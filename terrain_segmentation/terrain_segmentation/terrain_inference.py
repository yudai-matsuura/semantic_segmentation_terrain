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

class SegmentationNode(Node):
    def __init__(self):
        super().__init__('segmentation_node')
        # Publisher
        self.publisher_ = self.create_publisher(
            Image,
            '/segmentation/mask',
            10)
        # Subscriber
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/image_raw'
            self.image_callback,
            10)
        self.bridge = CvBridge()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)
        self.model.classifier[4] = nn.Conv2d(256, 2, kernel_size=1)
        self.model.aux_classifier[4] = nn.Conv2d(256, 2, kernel_size=1)

        model_path = '/home/go2laptop/yudai_ws/Inclination Terrain Segmentation.v1i.png-mask-semantic/deeplabv3_trained.pth'
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
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            pil_image = PILImage.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            input_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)['out']
                pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

            pred_resized = cv2.resize(pred.astype(np.uint8), (cv_image.shape[1], cv_image.shape[0], interpolation=cv2.INTER_NEAREST))
            color_mask = np.zeros_like(cv_image)
            color_mask[pred_resized == 1] = [0, 0, 255]

            mask_msg = self.bridge.cv2_to_imgmsg(color_mask, encoding='bgr8')
            mask_msg.header = msg.header
            self.publisher_.publish(mask_msg)

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