import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO


class RealSenseYOLO (Node):
    def __init__(self):
        super().__init__('realsense_yolo_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            'camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        self.model = YOLO("/home/go2laptop/yudai_ws/src/semantic_segmentation_terrain/scripts/runs/detect/train3/weights/best.pt")
        self.get_logger().info("YOLO model loaded successfully")
        self.output_path = "/home/go2laptop/yudai_ws/output_video.mp4"
        self.fps = 30
        self.out_writer = None

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            if self.out_writer is None:
                height, width, _ = cv_image.shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
                self.get_logger().info(f"Video writer initialized: {self.output_path}")

            results = self.model(cv_image, conf=0.05, iou=0.3)
            annotated = results[0].plot()

            self.out_writer.write(annotated)
            cv2.imshow("YOLOv8 Inference", annotated)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in callback: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = RealSenseYOLO()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
