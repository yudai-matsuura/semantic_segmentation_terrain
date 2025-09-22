import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as PILImage
import torch
from torchvision import models, transforms
import torch.nn as nn
import os


class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')
        # Publisher
        self.mask_publisher_ = self.create_publisher(
            Image,
            '/segmentation/mask',
            10)
        self.overlaid_publisher_ = self.create_publisher(
            Image,
            '/segmentation/overlaid_image',
            10)
        # Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/throttle/camera/color/image_raw',
            self.image_callback,
            10)
        # Parameters
        self.declare_parameter('target_width', 320)  # Default value
        self.declare_parameter('target_height', 240)  # Default value
        self.target_width = self.get_parameter('target_width').value
        self.target_height = self.get_parameter('target_height').value
        self.get_logger().info(f"Target mask resolution set to : {self.target_width}x{self.target_height}")

        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)
        NUM_CLASSES = 5  # Adjust based on your dataset
        self.model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
        self.model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

        model_path = '/home/srl-limb-ws4/yudai_ws/BASEPROD_trained.pth'
        if not os.path.exists(model_path):
            self.get_logger().error(f"Model file notn found: {model_path}")
            return
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device).eval()

        # preprocess
        self.preprocess = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.get_logger().info("Segmentation node initialized")

    def image_callback(self, msg):
        # self.get_logger().info(f"Received color with stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09f}")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            input_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

            # start_time = self.get_clock().now()

            with torch.no_grad():
                output = self.model(input_tensor)['out']
                pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

            if self.device.type == 'cuda':
                torch.cuda.synchronize()

            # end_time = self.get_clock().now()
            # duration = end_time - start_time
            # self.get_logger().info(f"Inference time: {duration.nanoseconds / 1e9:.4f} seconds")

            target_size = (self.target_width, self.target_height)
            pred_resized_for_pointcloud = cv2.resize(pred.astype(np.uint8), target_size, interpolation=cv2.INTER_NEAREST)
            # color_mask = np.zeros_like(cv_image)
            # color_mask[pred_resized == 1] = [0, 0, 255]
            color_map = np.array([
                [0, 0, 0],         # 0: background
                [0, 0, 255],       # 1: bed rock (赤)
                [0, 255, 0],       # 2: elevated bed rock (緑)
                [255, 0, 0],       # 3: soil (青)
                [0, 255, 255]      # 4: uneven terrain (黄)
            ], dtype=np.uint8)

            color_mask_for_pointcloud = color_map[pred_resized_for_pointcloud]

            # Mask image
            mask_msg = self.bridge.cv2_to_imgmsg(color_mask_for_pointcloud, encoding='bgr8')
            mask_msg.header = msg.header
            self.mask_publisher_.publish(mask_msg)

            # Generate mask for visualization
            color_image_size = (cv_image.shape[1], cv_image.shape[0])
            pred_resized_for_viz = cv2.resize(pred, color_image_size, interpolation=cv2.INTER_NEAREST)
            color_mask_for_viz = color_map[pred_resized_for_viz]

            # Overlaid image
            alpha = 0.6
            beta = 0.4
            overlaid_image = cv2.addWeighted(cv_image, alpha, color_mask_for_viz, beta, 0)
            overlaid_msg = self.bridge.cv2_to_imgmsg(overlaid_image, encoding='bgr8')
            overlaid_msg.header = msg.header
            self.overlaid_publisher_.publish(overlaid_msg)

        except Exception as e:
            self.get_logger().error(f"Falied to process image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()