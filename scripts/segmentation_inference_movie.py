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
    video_path = "/home/go2laptop/yudai_ws/video_name.mp4"
    output_path = "/home/go2laptop/yudai_ws"
    NUM_CLASSES = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.segmentation.deeplabv3_resnet50(weight=None, aux_loss=True)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.load_state_dict(torch.load("/path/to/deeplabv3_trained.pth", map_location=device))
    model = model.to(device).eval()
