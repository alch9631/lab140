import rclpy
import math
import signal
import cv2
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class ParcoursLineFollower(Node):
    def __init__(self):
        super().__init__('parcours_line_follower')
        
        self.bridge = CvBridge()
        
        # Publisher & Subscriber
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.img_sub = self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)
        
        # Zustand / Variablen
        self.camera_error = 0.0
        self.line_detected = False
        self.front_obstacle_dist = float('inf')

        # Geschwindigkeiten für den Parcours
        self.base_forward_speed = 0.35   # Konstante Vorwärtsgeschwindigkeit auf der Spur
        self.obstacle_stop_dist = 0.45   # Halteabstand vor Hindernissen
        
        # Kamera/Lenk-Parameter
        self.k_angular = 2.2             # Höherer Wert für schärfere Kurven im Parcours
        self.max_angular = 1.0           # Maximale Drehgeschwindigkeit
        self.roi_height = 50             # Bereich am unteren Bildrand

    def stop_robot(self):
        stop = Twist()
        for _ in range(10):
            self.cmd_pub.publish(stop)

    def scan_callback(self, scan):
        """ LiDAR filtert NUR den Bereich direkt vor dem Roboter (z.B. -30° bis +30°) """
        local_min_dist = float('inf')
        
        for i, r in enumerate(scan.ranges):
            if math.isnan(r) or math.isinf(r):
                continue
            if not (scan.range_min < r < scan.range_max):
                continue
            
            # Winkel berechnen
            angle = scan.angle_min + i * scan.angle_increment
            
            # WICHTIG: Nur Hindernisse im 60°-Kegel direkt vor dem Roboter beachten
            if abs(angle) < math.radians(30):
                if r < local_min_dist:
                    local_min_dist = r
                    
        self.front_obstacle_dist = local_min_dist

    def image_callback(self, msg):
        """ Verarbeitet das Bild und steuert DIREKT in Echtzeit (Parallel zum LiDAR) """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            height, width = cv_image.shape[:2]
            image_center = width // 2
            
            # Region of Interest (ROI) unten ausschneiden
            roi_top = height - self.roi_height
            roi = cv_image[roi_top:height, 0:width]
            
            # Graustufen und Binarisierung (Otsu wählt den Schwellenwert automatisch!)
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Konturen finden
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cmd = Twist()
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 150:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        
                        # Fehler berechnen (-1.0 bis 1.0)
                        self.camera_error = (cx - image_center) / image_center
                        self.line_detected = True
                        
                        # --- FAHRSTEUERUNG ---
                        # 1. Prüfen ob ein Hindernis VOR uns im Weg steht (LiDAR)
                        if self.front_obstacle_dist <= self.obstacle_stop_dist:
                            cmd.linear.x = 0.0  # Hindernis bremst uns aus
                            cmd.angular.z = 0.0
                        else:
                            # Freie Fahrt auf der Spur: Vorwärts fahren & Kamera lenkt
                            cmd.linear.x = self.base_forward_speed
                            raw_angular = -self.k_angular * self.camera_error
                            cmd.angular.z = max(-self.max_angular, min(raw_angular, self.max_angular))
                        
                        self.cmd_pub.publish(cmd)
                        return

            # Keine Linie gefunden -> Suchmodus (Langsam drehen auf der Stelle)
            self.line_detected = False
            cmd.linear.x = 0.0
            cmd.angular.z = 0.4
            self.cmd_pub.publish(cmd)
            
        except Exception as e:
            self.get_logger().error(f'Bildverarbeitungsfehler: {str(e)}')


def main():
    rclpy.init()
    node = ParcoursLineFollower()

    def emergency_stop(sig, frame):
        try:
            node.get_logger().info("Parcours gestoppt.")
            node.stop_robot()
        finally:
            node.destroy_node()
            rclpy.shutdown()

    signal.signal(signal.SIGINT, emergency_stop)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
