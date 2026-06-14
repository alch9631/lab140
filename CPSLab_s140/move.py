import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time
import signal

class ToggleDrive(Node):
    def __init__(self):
        super().__init__('toggle_drive_node')
        # Wichtig: Nutze exakt die Standard-QoS
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.move_forward = True

    def send_cmd(self, speed):
        msg = Twist()
        msg.linear.x = float(speed)
        self.publisher_.publish(msg)

def main():
    # TRICK: ROS sagen, dass es Signale NICHT selbst handhaben soll
    rclpy.init(signal_handler_options=rclpy.signals.SignalHandlerOptions.NO)
    
    node = ToggleDrive()
    print("Skript läuft. Beenden mit Strg+C...")

    try:
        last_toggle = time.time()
        while True:
            # Richtungswechsel-Logik
            if time.time() - last_toggle > 1.0:
                node.move_forward = not node.move_forward
                last_toggle = time.time()
                #print(f"Fahre {'VOR' if node.move_forward else 'ZURÜCK'}")

            node.send_cmd(0.5 if node.move_forward else -0.5)
            
            # Wichtig für die serielle Kommunikation zum ESP32
            rclpy.spin_once(node, timeout_sec=0.05)
            
    except KeyboardInterrupt:
        print("\n[STOP] Strg+C erkannt. Sende Stop-Befehl an ESP32...")
        
        # Hier ist der Kontext jetzt GARANTIERT noch gültig
        for _ in range(10):
            node.send_cmd(0.0)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.02)
        #print("[OK] Roboter sollte stehen.")

    finally:
        # Erst jetzt alles sauber beenden
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
