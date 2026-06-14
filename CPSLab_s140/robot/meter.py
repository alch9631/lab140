import rclpy, time
from rclpy.node import Node
from geometry_msgs.msg import Twist

DISTANCE = 1.0      # meters
SPEED = 0.2         # m/s
CALIBRATION = 1.25

class MoveOneMeter(Node):
   def __init__(self):
       super().__init__('move_one_meter')
       self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

       msg = Twist()
       msg.linear.x = SPEED

       duration = (DISTANCE / SPEED) * CALIBRATION  # time = d / v
       start = time.time()

       while time.time() - start < duration:
           self.pub.publish(msg)
           time.sleep(0.1)

       # stop
       msg.linear.x = 0.0
       self.pub.publish(msg)

rclpy.init()
node = MoveOneMeter()
rclpy.shutdown()
