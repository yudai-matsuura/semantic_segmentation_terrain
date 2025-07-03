import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
import torch.nn as nn
import cv2

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict(model, image, device):
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)['out']
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()
    return pred


def main():
    video_path = "/home/go2laptop/yudai_ws/for_annotation_1.mp4"
    output_path = "/home/go2laptop/yudai_ws"
    NUM_CLASSES = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.segmentation.deeplabv3_resnet50(weight=None, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.load_state_dict(torch.load("/home/go2laptop/yudai_ws/Inclination Terrain Segmentation.v1i.png-mask-semantic/deeplabv3_trained.pth", map_location=device))
    model = model.to(device).eval()

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    color_map = np.array([
        [0, 0, 0],
        [255, 0, 0]
    ])

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # OpenCV(BGR) → PIL(RGB)
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        pred_mask = predict(model, pil_image, device)
        pred_mask_resized = Image.fromarray(pred_mask.astype(np.uint8)).resize((width, height), resample=Image.NEAREST)
        color_mask = color_map[np.array(pred_mask_resized)]
        color_mask_bgr = cv2.cvtColor(color_mask.astype(np.uint8), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(frame, 0.5, color_mask_bgr, 0.5, 0)

        cv2.imshow('Segmentation Overlay', overlay)
        out.write(overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()