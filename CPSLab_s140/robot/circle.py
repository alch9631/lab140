

import rclpy, time, math
from geometry_msgs.msg import Twist

RADIUS = 0.3
SPEED = 0.3
CALIBRATION = 1.2

omega = SPEED / RADIUS

rclpy.init()
node = rclpy.create_node('circle_motion')
pub = node.create_publisher(Twist, '/cmd_vel', 10)

cmd = Twist()
cmd.linear.x = SPEED
cmd.angular.z = omega

# time to complete full circle
duration = (2 * math.pi * RADIUS) / SPEED * CALIBRATION

start = time.time()

while time.time() - start < duration:
   pub.publish(cmd)
   time.sleep(0.05)

# stop
cmd.linear.x = 0.0
cmd.angular.z = 0.0
pub.publish(cmd)

node.destroy_node()
rclpy.shutdown()

