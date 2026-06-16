

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2


CHECKERBOARD = (8, 6)        # innere Ecken (Felder - 1 je Richtung)
SQUARE_SIZE_MM = 25.0        # Kantenlaenge eines Feldes in mm
IN_TOPIC = "/image_raw/compressed"
OUT_TOPIC = "/calib/image/compressed"
TARGET_VIEWS = 20            # so viele Ansichten sammeln, dann kalibrieren
MIN_MOVE_PX = 40.0           # Brett muss sich so weit bewegt haben (Zentrum)
MIN_AREA_RATIO = 0.15        # ODER Flaeche so stark geaendert (Abstand)


CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def corner_stats(corners):
    pts = corners.reshape(-1, 2)
    center = pts.mean(axis=0)
    w = pts[:, 0].max() - pts[:, 0].min()
    h = pts[:, 1].max() - pts[:, 1].min()
    return center, float(w * h)


class HeadlessCalib(Node):
    def __init__(self):
        super().__init__("calib_headless")
        self.sub = self.create_subscription(
            CompressedImage, IN_TOPIC, self.cb, 10)
        self.pub = self.create_publisher(CompressedImage, OUT_TOPIC, 10)

        self.objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:CHECKERBOARD[0],
                                    0:CHECKERBOARD[1]].T.reshape(-1, 2)
        self.objp *= SQUARE_SIZE_MM

        self.objpoints, self.imgpoints = [], []
        self.accepted_stats = []          # (center, area) je akzeptierter Ansicht
        self.img_size = None
        self.done = False
        # set False if running headless / over SSH without X forwarding
        self.show_windows = True
        self.get_logger().info(
            f"Abonniere {IN_TOPIC}. Brett langsam vor der Kamera bewegen. "
            f"Ziel: {TARGET_VIEWS} Ansichten. Vorschau auf {OUT_TOPIC}.")

    def is_new_view(self, corners):
        center, area = corner_stats(corners)
        for c, a in self.accepted_stats:
            moved = np.linalg.norm(center - c)
            area_ratio = abs(area - a) / max(a, 1.0)
            if moved < MIN_MOVE_PX and area_ratio < MIN_AREA_RATIO:
                return False
        return True

    def publish_preview(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            return
        out = CompressedImage()
        out.header.stamp = self.get_clock().now().to_msg()
        out.format = "jpeg"
        out.data = buf.tobytes()
        self.pub.publish(out)

    def cb(self, msg):
        if self.done:
            return
        arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # robust detection: handles uneven lighting + skips frames with no board
        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                 + cv2.CALIB_CB_NORMALIZE_IMAGE
                 + cv2.CALIB_CB_FAST_CHECK)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        view = frame.copy()

        if found:
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), CRITERIA)
            cv2.drawChessboardCorners(view, CHECKERBOARD, corners2, found)
            if self.is_new_view(corners2):
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners2)
                self.accepted_stats.append(corner_stats(corners2))
                self.img_size = gray.shape[::-1]
                self.get_logger().info(
                    f"Ansicht {len(self.objpoints)}/{TARGET_VIEWS} gesammelt")

        # status banner: green = board seen this frame, red = not found
        status = "BOARD FOUND" if found else "NO BOARD"
        color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(view, status, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(view, f"{len(self.objpoints)}/{TARGET_VIEWS}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        self.publish_preview(view)

        # live popup window (set show_windows=False for headless/SSH)
        if self.show_windows:
            cv2.imshow("Calibration", view)
            cv2.waitKey(1)

        if len(self.objpoints) >= TARGET_VIEWS:
            self.done = True
            self.calibrate()

    def calibrate(self):
        n = len(self.objpoints)
        if n < 5:
            self.get_logger().warn(f"Nur {n} Ansichten - zu wenig.")
            return
        self.get_logger().info(f"Kalibriere mit {n} Ansichten ...")
        ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.img_size, None, None)

        total = 0.0
        for i in range(n):
            proj, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], K, dist)
            total += cv2.norm(self.imgpoints[i], proj, cv2.NORM_L2) / len(proj)
        mean_err = total / n

        self.get_logger().info(f"\nKameramatrix K:\n{K}")
        self.get_logger().info(f"Verzeichnung: {dist.ravel()}")
        self.get_logger().info(f"Reprojektionsfehler: {mean_err:.4f} px")

        fs = cv2.FileStorage("camera_calibration.yaml", cv2.FILE_STORAGE_WRITE)
        fs.write("image_width", self.img_size[0])
        fs.write("image_height", self.img_size[1])
        fs.write("camera_matrix", K)
        fs.write("distortion_coefficients", dist)
        fs.write("reprojection_error", mean_err)
        fs.release()
        self.get_logger().info("Gespeichert: camera_calibration.yaml")


def main():
    rclpy.init()
    node = HeadlessCalib()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().info("Abbruch - kalibriere mit bisherigen Ansichten.")
        node.calibrate()
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
