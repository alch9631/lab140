import rclpy
import cv2
import numpy as np
import time

from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32


class LineFollowerDebug(Node):
    def __init__(self):
        super().__init__('line_follower_debug')

        self.mask_pub = self.create_publisher(
            CompressedImage,
            '/line_mask/compressed',
            10
        )

        self.roi_pub = self.create_publisher(
            CompressedImage,
            '/line_roi/compressed',
            10
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.pan_pub = self.create_publisher(Int32, '/servo_s1', 10)
        self.tilt_pub = self.create_publisher(Int32, '/servo_s2', 10)

        self.image_sub = self.create_subscribtion (
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            10
        )

        self.pan_angle = 0
        self.tilt_angle = -100  # change if camera is not looking down

        self.speed = 0.08
        self.k_turn = 0.004
        self.max_turn = 0.6

        self.black_threshold = 70

        self.timer = self.create_timer(0.1, self.callback)

        self.get_logger().info('Line follower debug started')

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

    def callback(self):
        # fixed camera center
        self.pan_pub.publish(Int32(data=self.pan_angle))
        self.tilt_pub.publish(Int32(data=self.tilt_angle))

        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn('Camera frame not received')
            return

        h, w, _ = frame.shape

        # use only bottom half: floor area
        roi = frame[int(h * 0.55):h, :]

        self.publish_image(self.roi_pub, roi)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # black line -> white, everything else -> black
        mask = cv2.inRange(gray, 0, self.black_threshold)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        self.publish_image(self.mask_pub, mask)

        M = cv2.moments(mask)
        cmd = Twist()

        if M["m00"] > 1000:
            cx = int(M["m10"] / M["m00"])
            error = cx - (w // 2)

            cmd.linear.x = self.speed
            cmd.angular.z = -self.k_turn * error
            cmd.angular.z = max(-self.max_turn, min(self.max_turn, cmd.angular.z))

            self.get_logger().info(f'LINE FOUND cx={cx}, error={error}')
        else:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().warn('LINE NOT FOUND')

        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = LineFollowerDebug()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.stop_robot()
        node.cap.release()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()