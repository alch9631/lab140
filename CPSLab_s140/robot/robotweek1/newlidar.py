
import rclpy
import math
import signal
import sys
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class LidarDistanceKeeper360(Node):
    def __init__(self):
        super().__init__('lidar_distance_keeper_360')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        

        self.target_distance = 0.45
        self.deadband = 0.04

        self.k_linear = 0.9
        self.max_forward_speed = 0.75
        self.max_backward_speed = 0.55

        self.k_angular = 1.8
        self.max_angular = 1.2

    def stop_robot(self):
        stop = Twist()
        stop.linear.x = 0.0
        stop.angular.z = 0.0

        for _ in range(30):   # stronger immediate stop
            self.cmd_pub.publish(stop)

    def scan_callback(self, scan):
        best_range = float('inf')
        best_angle = 0.0

        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                continue

            if not (scan.range_min < r < scan.range_max):
                continue

            angle = scan.angle_min + i * scan.angle_increment

            # 360° detection: no front-angle filtering
            if r < best_range:
                best_range = r
                best_angle = angle

        cmd = Twist()

        if best_range == float('inf'):
            self.cmd_pub.publish(cmd)
            return

        error = best_range - self.target_distance

        if abs(error) <= self.deadband:
            cmd.linear.x = 0.0
        else:
            speed = self.k_linear * error

            if speed > 0:
                cmd.linear.x = min(speed, self.max_forward_speed)
            else:
                cmd.linear.x = max(speed, -self.max_backward_speed)

        cmd.angular.z = self.k_angular * best_angle
        cmd.angular.z = max(-self.max_angular, min(cmd.angular.z, self.max_angular))

        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = LidarDistanceKeeper360()
    def emergency_stop(sig, frame):
        try:
            stop_msg = Twist()
            node.cmd_pub.publish(stop_msg)

        finally:
            node.destroy_node()
            rclpy.shutdown()

    # Registriere den Notstopp beim Betriebssystem
    signal.signal(signal.SIGINT, emergency_stop)

    while rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)


if __name__ == '__main__':
    main()