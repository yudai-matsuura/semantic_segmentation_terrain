import rclpy
from rclpy.serialization import serialize_message
from builtin_interfaces.msg import Time as TimeMsg
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import rosbag2_py
from pathlib import Path
import cv2
import numpy as np
import re

# --- 設定 ---
RGB_DIR   = Path("/media/go2laptop/T7/Moon_9/image0/color")
DEPTH_DIR = Path("/media/go2laptop/T7/Moon_9/image0/depth")
LABEL_DIR = Path("/media/go2laptop/T7/Moon_9/image0/label")

OUTPUT_BAG = Path("/media/go2laptop/T7/Moon_9_rosbag/lusnar_all_bag")
FRAME_ID = "camera/optical/frame"

TOPIC_RGB   = "/rgb/image_raw"
TOPIC_DEPTH = "/depth/image_raw"
TOPIC_LABEL = "/label/image_raw"

# --- PFM読み込み ---
def load_pfm(file_path):
    with open(file_path, 'rb') as f:
        header = f.readline().rstrip().decode('utf-8')
        if header not in ['PF', 'Pf']:
            raise Exception('Not a PFM file.')

        color = header == 'PF'
        dim_line = f.readline().decode('utf-8')
        while dim_line.startswith('#'):  # コメント行スキップ
            dim_line = f.readline().decode('utf-8')
        width, height = map(int, dim_line.strip().split())

        scale = float(f.readline().decode('utf-8').strip())
        endian = '<' if scale < 0 else '>'
        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flipud(data)
        return data

# --- 初期化 ---
rclpy.init()
bridge = CvBridge()

storage_options = rosbag2_py.StorageOptions(uri=str(OUTPUT_BAG), storage_id='sqlite3')
converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
writer = rosbag2_py.SequentialWriter()
writer.open(storage_options, converter_options)

# トピック作成
for topic_name in [TOPIC_RGB, TOPIC_DEPTH, TOPIC_LABEL]:
    writer.create_topic(rosbag2_py.TopicMetadata(
        name=topic_name,
        type='sensor_msgs/msg/Image',
        serialization_format='cdr'
    ))

# --- ファイル一覧取得（共通のタイムスタンプに基づいて） ---
timestamps = sorted([
    p.stem for p in RGB_DIR.glob("*.png")
    if p.stem.isdigit()
])

if not timestamps:
    print(f"[ERROR] RGB画像が見つかりません: {RGB_DIR}")
    exit(1)

for ts in timestamps:
    try:
        sec = int(ts) // int(1e9)
        nanosec = int(ts) % int(1e9)

        header = Header()
        header.stamp = TimeMsg(sec=sec, nanosec=nanosec)
        header.frame_id = FRAME_ID

        # RGB
        rgb_path = RGB_DIR / f"{ts}.png"
        rgb_img = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_img is not None:
            msg = bridge.cv2_to_imgmsg(rgb_img, encoding="bgr8")
            msg.header = header
            writer.write(TOPIC_RGB, serialize_message(msg), sec * 1_000_000_000 + nanosec)
        else:
            print(f"[WARN] RGB 読み込み失敗: {rgb_path}")

        # Depth
        depth_path = DEPTH_DIR / f"{ts}.pfm"
        if depth_path.exists():
            depth_np = load_pfm(depth_path)
            depth_mm = (depth_np * 1000).astype(np.uint16)
            msg = bridge.cv2_to_imgmsg(depth_mm, encoding="16UC1")
            msg.header = header
            writer.write(TOPIC_DEPTH, serialize_message(msg), sec * 1_000_000_000 + nanosec)
        else:
            print(f"[WARN] Depth 読み込み失敗: {depth_path}")

        # Label
        label_path = LABEL_DIR / f"{ts}.png"
        label_img = cv2.imread(str(label_path), cv2.IMREAD_COLOR)
        if label_img is not None:
            msg = bridge.cv2_to_imgmsg(label_img, encoding="bgr8")
            msg.header = header
            writer.write(TOPIC_LABEL, serialize_message(msg), sec * 1_000_000_000 + nanosec)
        else:
            print(f"[WARN] Label 読み込み失敗: {label_path}")

    except Exception as e:
        print(f"[ERROR] {ts} の処理中にエラー: {e}")

print(f"✅ 完了：{len(timestamps)}フレームを書き込みました → {OUTPUT_BAG}")
rclpy.shutdown()
