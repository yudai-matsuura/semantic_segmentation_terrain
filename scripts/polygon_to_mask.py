import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
from tqdm import tqdm

json_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.json')
img_dir = "your_dataset/images"
mask_output_dir = "your_dataset/masks"

coco = COCO(json_path)
img_ids = coco.getImgIds()

for img_id in tqdm(img_ids):
    img_info = coco.loadImgs(img_id)[0]
    file_name = img_info['file_name']
    height, width = img_info['height'], img_info['width']

    mask = np.zeros((height, width), dtype=np.uint8)
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        cat_id = ann['category_id']
        segmentation = ann['segmentation']

        if isinstance(segmentation, list):  # polygon
            for seg in segmentation:
                polygon = np.array(seg).reshape((-1, 2)).astype(np.int32)
                cv2.fillPoly(mask, [polygon], color=cat_id)

    out_path = os.path.join(mask_output_dir, file_name.replace('.jpg', '.png'))
    Image.fromarray(mask).save(out_path)