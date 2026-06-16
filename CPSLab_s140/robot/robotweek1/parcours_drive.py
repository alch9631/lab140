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
    """Autonomous parcours driver.

    - BLACK path  -> follow it (primary steering, centroid + P controller)
    - WHITE edges -> track boundary, steer away to stay inside
    - GREEN area  -> finish/goal zone, stop and latch
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
        self.tilt_angle = -100   # change if camera is not looking down

        # --- driving params ---
        self.speed = 0.12
        self.k_turn = 0.004      # steering gain on black-line pixel error
        self.max_turn = 0.6
        self.k_white = 0.6       # steering gain to push away from white edge

        # --- detection thresholds ---
        self.black_threshold = 70    # gray <= this -> black line
        self.white_threshold = 185   # gray >= this -> white edge
        self.green_stop_ratio = 0.12  # green fraction of ROI -> finish, stop
        # green in HSV (OpenCV H is 0..179)
        self.green_lo = np.array([40, 60, 50], np.uint8)
        self.green_hi = np.array([85, 255, 255], np.uint8)

        # state
        self.finished = False

        # auto-headless: only show windows if a display is available
        self.show_windows = bool(os.environ.get('DISPLAY'))

        self.get_logger().info(
            f'Parcours drive started (windows={self.show_windows})')

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

    def image_callback(self, msg):
        self.pan_pub.publish(Int32(data=self.pan_angle))
        self.tilt_pub.publish(Int32(data=self.tilt_angle))

        buf = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            self.get_logger().warn('Could not decode camera frame')
            return

        h, w, _ = frame.shape
        roi = frame[int(h * 0.55):h, :]           # floor in front of robot
        rh, rw, _ = roi.shape

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # --- GREEN finish zone ---
        green = cv2.inRange(hsv, self.green_lo, self.green_hi)
        green_ratio = float(cv2.countNonZero(green)) / (rh * rw)

        # --- BLACK line ---
        black = cv2.inRange(gray, 0, self.black_threshold)
        kernel = np.ones((5, 5), np.uint8)
        black = cv2.morphologyEx(black, cv2.MORPH_OPEN, kernel)
        black = cv2.morphologyEx(black, cv2.MORPH_CLOSE, kernel)

        # --- WHITE edges (left vs right presence) ---
        white = cv2.inRange(gray, self.white_threshold, 255)
        margin = rw // 3
        left_white = float(cv2.countNonZero(white[:, :margin])) / (rh * margin)
        right_white = float(cv2.countNonZero(white[:, -margin:])) / (rh * margin)

        cmd = Twist()
        cx = None
        state = ''

        if self.finished or green_ratio >= self.green_stop_ratio:
            # GREEN finish zone -> stop and latch
            self.finished = True
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            state = 'FINISH (green) - STOPPED'
        else:
            M = cv2.moments(black)
            if M['m00'] > 1000:
                # follow the black line
                cx = int(M['m10'] / M['m00'])
                error = cx - (rw // 2)
                ang = -self.k_turn * error
                # add a gentle push away from the nearer white edge
                ang += -self.k_white * (left_white - right_white)
                cmd.linear.x = self.speed
                cmd.angular.z = max(-self.max_turn, min(self.max_turn, ang))
                state = f'LINE cx={cx} err={error} wL={left_white:.2f} wR={right_white:.2f}'
            elif left_white > 0.2 or right_white > 0.2:
                # no line but boundary visible -> steer toward open side
                cmd.linear.x = self.speed * 0.5
                cmd.angular.z = -self.k_white * (left_white - right_white)
                cmd.angular.z = max(-self.max_turn, min(self.max_turn, cmd.angular.z))
                state = f'BOUNDARY wL={left_white:.2f} wR={right_white:.2f}'
            else:
                # nothing -> search in place
                cmd.linear.x = 0.0
                cmd.angular.z = 0.4
                state = 'SEARCHING (no line)'

        self.cmd_pub.publish(cmd)
        self.get_logger().info(state)

        # --- debug output ---
        overlay = roi.copy()
        cv2.line(overlay, (rw // 2, 0), (rw // 2, rh), (255, 0, 0), 1)
        if cx is not None:
            cv2.line(overlay, (cx, 0), (cx, rh), (0, 0, 255), 2)
        col = (0, 255, 0) if not self.finished else (0, 0, 255)
        cv2.putText(overlay, state[:42], (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.putText(overlay, f'green={green_ratio:.2f}', (5, rh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 0), 1)

        # combined mask preview: black=R, white=B, green=G
        masks = np.zeros_like(roi)
        masks[:, :, 2] = black
        masks[:, :, 0] = white
        masks[:, :, 1] = green

        self.publish_image(self.overlay_pub, overlay)
        self.publish_image(self.mask_pub, masks)

        if self.show_windows:
            cv2.imshow('Parcours (overlay)', overlay)
            cv2.imshow('Masks B=black G=green Bl=white', masks)
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
