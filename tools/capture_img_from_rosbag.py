import pathlib
from datetime import datetime

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')

        default_topic = '/throttle/camera/color/image_raw'
        default_output_dir = '/media/go2laptop/T7/dataset_training/raw_color_images'

        self.declare_parameter('topic', default_topic)
        self.declare_parameter('output_dir', default_output_dir)

        topic = self.get_parameter('topic').get_parameter_value().string_value
        output_dir = pathlib.Path(
            self.get_parameter('output_dir').get_parameter_value().string_value
            ).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir

        # Subscriber
        self.sub = self.create_subscription(Image, topic, self.callback, 10)
        self.bridge = CvBridge()
        self.last_frame = None
        self.get_logger().info(
            f'Listening to "{topic}". Press SPACE in the image window to save.'
        )

    def callback(self, msg: Image):
        # Convert to ROS message
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.last_frame = cv_img

        # Show preview
        cv2.imshow('image_raw (press SPACE to save)', cv_img)
        key = cv2.waitkey(1) & 0xFF

        # Save image
        if key == 32:
            stamp = msg.header.stamp
            ts = datetime.fromtimestamp(stamp.sec + stamp.nanosec * 1e-9)
            filename = self.output_dir / f'{ts:%Y%m%d_%H%M%S_%f}.png'
            cv2.imwrite(str(filename), cv_img)
            self.get_logger().info(f'Saved: {filename}')


def main(args=None):
    rclpy.init(args=args)
    node = ImageSaver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
