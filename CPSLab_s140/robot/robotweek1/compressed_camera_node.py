import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
import cmd
class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.pub = self.create_publisher(CompressedImage, '/image_raw/compressed', 10)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(20,cv2.CAP_V4L2)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.timer = self.create_timer(0.1, self.callback)
        self.get_logger().info('Camera node started')

    def callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.format = 'jpeg'
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            msg.data = buf.tobytes()
            self.pub.publish(msg)
            
        
       #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)#grayscale image
        #oi = gray[300:480, :] #crops the image
       #_, mask = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY_INV)#binary threshold
       #M = cv2.moments(mask)
       #if M["m00"] != 0:
        #   cx = int(M["m10"] / M["m00"])#center of moments filter
         #  width = mask.shape[1]
          # error = cx - width //2 
           #self.error = error


rclpy.init()
node = CameraNode()
rclpy.spin(node)