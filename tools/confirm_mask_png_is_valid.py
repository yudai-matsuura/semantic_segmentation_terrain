import cv2
import numpy as np

# 真っ黒に見えるマスク画像のパスを指定
mask_image = cv2.imread('/home/go2laptop/yudai_ws/Inclination Terrain Segmentation.v1i.png-mask-semantic/test/frame_0090_png.rf.49c9d4c20b9dc3c41b2503e332799c1e_mask.png', cv2.IMREAD_GRAYSCALE)

# 画像に含まれるユニークなピクセル値を確認
unique_values = np.unique(mask_image)

print(f"マスク画像に含まれるピクセル値: {unique_values}")

# => マスク画像に含まれるピクセル値: [0] と出たら、アノテーションがない可能性が高い
# => マスク画像に含まれるピクセル値: [0 1 2] のように出たら、見た目が黒いだけでデータは正常