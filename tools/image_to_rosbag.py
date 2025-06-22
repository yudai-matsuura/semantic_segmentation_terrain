import rclpy
from rclpy.serialization import serialize_message
from builtin_interfaces.msg import Time as TimeMsg # 衝突を避けるためエイリアス
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import rosbag2_py
from pathlib import Path
import cv2

# --- config ---
RGB_DIR = Path("/media/go2laptop/T7/Moon_9/image0/color")
OUTPUT_BAG = Path("/media/go2laptop/T7/Moon_9_rosbag/lusnar_rgb_bag")
TOPIC_NAME = "/rgb/image_raw"
FRAME_ID = "camera_rgb_optical_frame"

# --- init ---
# rclpy.init() と rclpy.shutdown() は main 関数内で呼び出すのがベストプラクティス
# ただし、単一のスクリプトとして動かす分には、このように記述してもエラーにはなりにくい
rclpy.init(args=None)
bridge = CvBridge()

# Bag Writer
storage_options = rosbag2_py.StorageOptions(uri=str(OUTPUT_BAG), storage_id='sqlite3') # Pathオブジェクトをstrに変換
converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
writer = rosbag2_py.SequentialWriter()

try:
    writer.open(storage_options, converter_options)
except Exception as e:
    print(f"[ERROR] Bag Writerを開けませんでした: {e}")
    rclpy.shutdown()
    exit(1)

# make topic
writer.create_topic(rosbag2_py.TopicMetadata(
    name=TOPIC_NAME,
    type='sensor_msgs/msg/Image',
    serialization_format='cdr'
))

image_paths = sorted([
    p for p in RGB_DIR.glob("*.png")
    if p.stem.isdigit()
])

if not image_paths:
    print(f"[ERROR] 指定されたディレクトリにPNG画像ファイルが見つかりませんでした: {RGB_DIR}")
    rclpy.shutdown()
    exit(1)

for img_path in image_paths:
    timestamp_ns = int(img_path.stem)
    sec = timestamp_ns // int(1e9)
    nanosec = timestamp_ns % int(1e9)

    cv_img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if cv_img is None:
        print(f"[WARN] 画像読み込み失敗: {img_path}")
        continue

    # Convert to ROS2 message
    img_msg = bridge.cv2_to_imgmsg(cv_img, encoding="bgr8")
    img_msg.header = Header()
    img_msg.header.stamp = TimeMsg(sec=sec, nanosec=nanosec)
    img_msg.header.frame_id = FRAME_ID

    # ROSbag2のwriter.writeはナノ秒単位の整数タイムスタンプを期待します
    timestamp_to_write = img_msg.header.stamp.sec * 1_000_000_000 + img_msg.header.stamp.nanosec
    writer.write(TOPIC_NAME, serialize_message(img_msg), timestamp_to_write)

print(f"✅ 完了：{len(image_paths)}枚のRGB画像を '{OUTPUT_BAG}/' にbag形式で保存しました。")
rclpy.shutdown()