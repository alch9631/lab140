import rclpy
import math
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from time import sleep


class LidarObjectFollower(Node):
    def __init__(self):
        super().__init__('lidar_object_follower')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.target_distance = 0.45

        # Faster movement
        self.max_speed = 0.45
        self.min_speed = 0.10
        self.k_linear = 0.85

        # Turning
        self.k_angular = 1.8
        self.max_angular = 1.2

        self.front_angle = math.radians(70)  # search ±70 degrees in front

    def stop_robot(self):
        stop = Twist()
        for _ in range(10):
            self.cmd_pub.publish(stop)

    def scan_callback(self, scan):
        best_range = float('inf')
        best_angle = 0.0

        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                continue

            angle = scan.angle_min + i * scan.angle_increment

            if abs(angle) > self.front_angle:
                continue

            if scan.range_min < r < scan.range_max and r < best_range:
                best_range = r
                best_angle = angle

        cmd = Twist()

        if best_range == float('inf'):
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
        else:
            distance_error = best_range - self.target_distance

            if distance_error > 0.03:
                speed = self.k_linear * distance_error
                cmd.linear.x = max(self.min_speed, min(speed, self.max_speed))
            else:
                cmd.linear.x = 0.0

            turn = self.k_angular * best_angle
            cmd.angular.z = max(-self.max_angular, min(turn, self.max_angular))

        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = LidarObjectFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()