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
        # binary white-line view (black/white) for debugging the wall detection
        self.white_pub = self.create_publisher(
            CompressedImage, '/parcours/white/compressed', 10)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # keep camera centered / looking down
        self.pan_pub = self.create_publisher(Int32, '/servo_s1', 10)
        self.tilt_pub = self.create_publisher(Int32, '/servo_s2', 10)

        self.image_sub = self.create_subscription(
            CompressedImage, '/image_raw/compressed', self.image_callback, 10)

        # camera angle (override per run, e.g. TILT=0 for straight-ahead view)
        self.pan_angle = int(os.environ.get('PAN', 0))
        self.tilt_angle = int(os.environ.get('TILT', -130))

        # --- driving params ---
        self.speed = 0.12
        self.k_turn = 1.3        # steering gain on normalized road error (-1..1)
        self.max_turn = 1.0      # allow sharp turns
        self.k_green = 0.6       # extra push away from green side
        self.k_wall = 0.35       # push away from the closer white line (wall)

        # --- ROI: ignore the top (room/horizon). Smaller = keep more of the
        # image, needed when the camera looks straight ahead (road sits higher).
        # Override per run with ROI_TOP=0.25 ... 0.5
        self.roi_top_frac = float(os.environ.get('ROI_TOP', 0.35))
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

        # pick ONE road blob. Prefer the one anchored at the bottom (road under
        # the robot) so it can't jump to a detached grey patch in the
        # background. If nothing reaches the bottom (camera looking straight
        # ahead -> road sits higher in the frame), fall back to the largest
        # blob so it still drives.
        num, labels = cv2.connectedComponents(road)
        road_clean = np.zeros_like(road)
        if num > 1:
            strip = labels[int(rh * 0.82):, :]       # bottom 18% of the ROI
            fg = strip[strip > 0]
            if fg.size > 50:
                vals, counts = np.unique(fg, return_counts=True)
                keep = int(vals[np.argmax(counts)])   # main blob under robot
            else:
                sizes = np.bincount(labels.ravel())
                sizes[0] = 0                          # ignore background
                keep = int(np.argmax(sizes))          # largest blob anywhere
            road_clean[labels == keep] = 255
        road = road_clean

        # green presence on left vs right (to push off-track edge away)
        half = rw // 2
        green_l = float(cv2.countNonZero(green[:, :half]))
        green_r = float(cv2.countNonZero(green[:, half:]))
        green_total = (green_l + green_r) / (rh * rw)
        green_bias = self.k_green * (green_r - green_l) / (rh * rw)

        # CENTERLINE between the two white lines: for several scan rows across
        # the lookahead band, take the midpoint between the left and right edge
        # of the grey corridor (= midway between the white walls). Average them
        # into one target. Nearer rows weigh a little more.
        la_top = int(rh * self.look_top)
        la_bot = int(rh * self.look_bot)
        full_pixels = cv2.countNonZero(road)

        centers = []
        cweights = []
        for y in np.linspace(la_top, la_bot - 1, 12).astype(int):
            xs = np.where(road[y] > 0)[0]
            if xs.size > 5:
                centers.append(0.5 * (float(xs[0]) + float(xs[-1])))
                cweights.append(0.5 + 0.5 * (y / max(rh - 1, 1)))

        cmd = Twist()
        cx = None
        state = ''

        if len(centers) >= 3:
            # follow the corridor centerline
            cx = float(np.average(centers, weights=np.array(cweights)))
            error = (cx - rw / 2.0) / (rw / 2.0)        # -1..1

            # white lines = hard walls: in the near band, find the closest
            # white line left and right of center; push away from the nearer.
            wall = 0.0
            nb = white[int(rh * 0.5):, :]
            cols = np.where(nb.sum(axis=0) > 0)[0]
            if cols.size:
                left = cols[cols < rw / 2]
                right = cols[cols > rw / 2]
                if left.size and right.size:
                    dist_l = rw / 2 - float(left.max())   # gap to left wall
                    dist_r = float(right.min()) - rw / 2  # gap to right wall
                    wall = -self.k_wall * (dist_r - dist_l) / (rw / 2)

            turn = -self.k_turn * error + green_bias + wall
            turn = max(-self.max_turn, min(self.max_turn, turn))
            # slow down in sharp curves so it can physically make the turn
            speed = self.speed * (1.0 - 0.6 * min(1.0, abs(error)))
            cmd.linear.x = max(0.04, speed)
            cmd.angular.z = turn
            self.last_turn = turn
            state = (f'CENTER err={error:+.2f} turn={turn:+.2f} '
                     f'v={cmd.linear.x:.2f} rows={len(centers)}')
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
        self.publish_image(self.white_pub, white)   # binary white-line view

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
