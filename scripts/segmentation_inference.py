import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
import torch.nn as nn

preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def predict(model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out']  # [1, NUM_CLASSES, H, W]
        pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    return pred, image


def main():
    image_path = "/home/go2laptop/yudai_ws/src/semantic_segmentation_terrain/data/train_images/frame_0104.png"
    NUM_CLASSES = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=True)

    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)

    model.load_state_dict(torch.load("/home/go2laptop/yudai_ws/Inclination Terrain Segmentation.v1i.png-mask-semantic/deeplabv3_trained.pth", map_location=device))

    model = model.to(device).eval()

    pred_mask, original_image = predict(model, image_path, device)
    pred_mask_resized = Image.fromarray(pred_mask.astype(np.uint8)).resize(original_image.size, resample=Image.NEAREST)

    color_map = np.array([
        [0, 0, 0],       # background
        [255, 0, 0]      # inclination_terrain
    ])
    color_mask = color_map[pred_mask_resized]

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image)
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(color_mask)
    plt.show()


if __name__ == "__main__":
    main()