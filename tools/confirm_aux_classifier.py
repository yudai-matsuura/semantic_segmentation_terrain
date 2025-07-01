import torch

state_dict = torch.load("/home/go2laptop/yudai_ws/src/semantic_segmentation_terrain/scripts/deeplabv3_trained.pth", map_location="cpu")

for name, param in state_dict.items():
    print(f"{name}: {param.shape}")
