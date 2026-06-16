import os
import time

import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32


class ParcoursDrive(Node):
    """Autonomous road follower.

    The track is a GREY road bordered by WHITE lines, with GREEN runoff
    outside. There is no line to follow - instead we track the grey road
    region itself and keep it centered, steering away from green/white.
    The dark room above the road (horizon) is cropped out by the ROI.
    """

    def __init__(self):
        super().__init__('parcours_drive')

        # debug image topics (view with rqt_image_view if headless)
        self.overlay_pub = self.create_publisher(
            CompressedImage, '/parcours/overlay/compressed', 10)
        self.mask_pub = self.create_publisher(
            CompressedImage, '/parcours/mask/compressed', 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # keep camera centered / looking down
        self.pan_pub = self.create_publisher(Int32, '/servo_s1', 10)
        self.tilt_pub = self.create_publisher(Int32, '/servo_s2', 10)

        self.image_sub = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.image_callback, 10)

        self.pan_angle = 0
        self.tilt_angle = -130   # tilt down so road fills the lower frame

        # --- driving params ---
        self.speed = 0.12
        self.k_turn = 1.3        # steering gain on normalized road error (-1..1)
        self.max_turn = 1.0      # allow sharp turns
        self.k_green = 0.6       # extra push away from green side

        # --- ROI: ignore the top (room/horizon) ---
        self.roi_top_frac = 0.45    # use everything below 45% of the height
        # lookahead band inside the ROI (fraction of ROI height) used to steer;
        # higher (toward 0) = looks farther ahead = reacts to curves sooner
        self.look_top = 0.05
        self.look_bot = 0.45

        # --- GREY ROAD in HSV: low saturation, mid brightness ---
        self.road_s_max = 55     # road is greyish -> low saturation
        self.road_v_min = 45     # brighter than dark clutter
        self.road_v_max = 170    # darker than white lines

        # --- WHITE edge: low saturation, very bright ---
        self.white_v_min = 175

        # --- GREEN runoff in HSV (OpenCV H is 0..179) ---
        self.green_lo = np.array([35, 50, 40], np.uint8)
        self.green_hi = np.array([95, 255, 255], np.uint8)

        # --- camera calibration: undistort the fisheye before processing ---
        self.K = None
        self.dist = None
        self.map1 = None
        self.map2 = None
        calib_path = os.environ.get('CALIB_FILE', 'camera_calibration.yaml')
        if not os.path.isfile(calib_path):
            here = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'camera_calibration.yaml')
            if os.path.isfile(here):
                calib_path = here
        fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
        if fs.isOpened():
            self.K = fs.getNode('camera_matrix').mat()
            self.dist = fs.getNode('distortion_coefficients').mat()
            fs.release()
            self.get_logger().info(f'Loaded calibration from {calib_path}')
        else:
            self.get_logger().warn(
                f'No calibration file ({calib_path}); running on raw frames')

        # last steering, used when the road is briefly lost
        self.last_turn = 0.0

        # auto-headless: only show windows if a display is available
        self.show_windows = bool(os.environ.get('DISPLAY'))

        self.get_logger().info(
            f'Parcours road follower started (windows={self.show_windows})')

    def stop_robot(self):
        stop = Twist()
        for _ in range(30):
            self.cmd_pub.publish(stop)
            time.sleep(0.02)

    def publish_image(self, pub, image):
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = 'jpeg'
        ok, buf = cv2.imencode('.jpg', image)
        if ok:
            msg.data = buf.tobytes()
            pub.publish(msg)

    def undistort(self, frame):
        """Remove fisheye distortion using the loaded calibration."""
        if self.K is None:
            return frame
        h, w = frame.shape[:2]
        if self.map1 is None:
            newK, _ = cv2.getOptimalNewCameraMatrix(
                self.K, self.dist, (w, h), 0, (w, h))
            self.map1, self.map2 = cv2.initUndistortRectifyMap(
                self.K, self.dist, None, newK, (w, h), cv2.CV_16SC2)
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

    def image_callback(self, msg):
        self.pan_pub.publish(Int32(data=self.pan_angle))
        self.tilt_pub.publish(Int32(data=self.tilt_angle))

        buf = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn('Could not decode camera frame')
            return

        frame = self.undistort(frame)

        h, w, _ = frame.shape
        roi = frame[int(h * self.roi_top_frac):h, :]   # drop the room/horizon
        rh, rw, _ = roi.shape

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # --- masks ---
        road = ((S <= self.road_s_max) &
                (V >= self.road_v_min) &
                (V <= self.road_v_max)).astype(np.uint8) * 255
        white = ((S <= 60) & (V >= self.white_v_min)).astype(np.uint8) * 255
        green = cv2.inRange(hsv, self.green_lo, self.green_hi)

        # clean the road mask
        kernel = np.ones((5, 5), np.uint8)
        road = cv2.morphologyEx(road, cv2.MORPH_OPEN, kernel)
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel)

        # keep only the road blob ANCHORED to the bottom of the ROI (the road
        # is always directly under the robot). This stops it from locking onto
        # a detached grey patch in the background and suddenly going straight.
        num, labels = cv2.connectedComponents(road)
        road_clean = np.zeros_like(road)
        if num > 1:
            strip = labels[int(rh * 0.85):, :]       # bottom 15% of the ROI
            fg = strip[strip > 0]
            if fg.size > 0:
                vals, counts = np.unique(fg, return_counts=True)
                keep = int(vals[np.argmax(counts)])   # main blob under robot
                road_clean[labels == keep] = 255
        road = road_clean

        # green presence on left vs right (to push off-track edge away)
        half = rw // 2
        green_l = float(cv2.countNonZero(green[:, :half]))
        green_r = float(cv2.countNonZero(green[:, half:]))
        green_total = (green_l + green_r) / (rh * rw)
        green_bias = self.k_green * (green_r - green_l) / (rh * rw)

        # LOOKAHEAD band: steer by where the road is FARTHER ahead, not the
        # near region. The near road barely shifts on a curve, which is why it
        # was understeering and going straight. The far band swings hard.
        la_top = int(rh * self.look_top)
        la_bot = int(rh * self.look_bot)
        band = road[la_top:la_bot, :]
        band_pixels = cv2.countNonZero(band)
        full_pixels = cv2.countNonZero(road)

        cmd = Twist()
        cx = None
        state = ''

        if band_pixels > 0.03 * (la_bot - la_top) * rw:
            # road visible far ahead -> normal lookahead steering
            M = cv2.moments(band)
            cx = M['m10'] / M['m00']
            error = (cx - rw / 2.0) / (rw / 2.0)        # -1..1
            turn = -self.k_turn * error + green_bias
            turn = max(-self.max_turn, min(self.max_turn, turn))
            # slow down in sharp curves so it can physically make the turn
            speed = self.speed * (1.0 - 0.6 * min(1.0, abs(error)))
            cmd.linear.x = max(0.04, speed)
            cmd.angular.z = turn
            self.last_turn = turn
            state = (f'ROAD err={error:+.2f} turn={turn:+.2f} '
                     f'v={cmd.linear.x:.2f} green={green_total:.2f}')
        elif full_pixels > 0.05 * rh * rw:
            # road only close (sharp curve leaving the far view) -> turn hard
            # toward whichever side the near road sits, crawl forward
            M = cv2.moments(road)
            cx = M['m10'] / M['m00']
            error = (cx - rw / 2.0) / (rw / 2.0)
            turn = -self.k_turn * 1.5 * error + green_bias
            turn = max(-self.max_turn, min(self.max_turn, turn))
            cmd.linear.x = self.speed * 0.35
            cmd.angular.z = turn
            self.last_turn = turn
            state = f'SHARP CURVE err={error:+.2f} turn={turn:+.2f}'
        else:
            # road lost -> creep and keep turning the last direction to recover
            cmd.linear.x = 0.0
            cmd.angular.z = 0.6 if self.last_turn >= 0 else -0.6
            state = f'NO ROAD - recovering (green={green_total:.2f})'

        self.cmd_pub.publish(cmd)
        self.get_logger().info(state)

        # --- debug output ---
        overlay = roi.copy()
        cv2.line(overlay, (rw // 2, 0), (rw // 2, rh), (255, 0, 0), 1)
        if cx is not None:
            cv2.line(overlay, (int(cx), 0), (int(cx), rh), (0, 0, 255), 2)
        cv2.putText(overlay, state[:46], (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # combined mask preview: road=grey, white=blue, green=green
        masks = np.zeros_like(roi)
        masks[road > 0] = (128, 128, 128)
        masks[white > 0] = (255, 0, 0)
        masks[green > 0] = (0, 255, 0)

        self.publish_image(self.overlay_pub, overlay)
        self.publish_image(self.mask_pub, masks)

        if self.show_windows:
            cv2.imshow('Parcours (overlay)', overlay)
            cv2.imshow('Masks: grey=road blue=white green=green', masks)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = ParcoursDrive()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
