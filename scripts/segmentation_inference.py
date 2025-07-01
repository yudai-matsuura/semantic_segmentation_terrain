import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader


# ===== Dataset Class =====
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_idr = image_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png') and "_mask" not in f])
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = img_path.replace(".png", "_mask.png")
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        mask = np.array(mask, dtype=np.int64)

        return image, torch.tensor(mask, dtype=torch.long)


# ===== Train Function =====
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(dataloader):
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        return total_loss / len(dataloader)


# ===== Main =====
def main():
    train_dir = "/home/go2laptop/yudai_ws/Inclination Terrain Segmentation.v1i.png-mask-semantic/train"
    val_dir = "/home/go2laptop/yudai_ws/Inclination Terrain Segmentation.v1i.png-mask-semantic/valid"

    NUM_CLASSES = 2  # 0: background, 1: inclination_terrain

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = SegmentationDataset(train_dir, transform=transform)
    val_dataset = SegmentationDataset(val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch+1} - Train Loss: {loss:.4f}")

    torch.save(model.state_dict(), "deeplabv3_trained.pth")
    print("✅ モデル保存完了: deeplabv3_trained.pth")


if __name__ == "__main__":
    main()
