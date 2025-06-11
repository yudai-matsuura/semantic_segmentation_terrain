from ultralytics import YOLO

if __name__ == "__main__":
    mode1 = YOLO('yolov8n.pt')
    mode1.train(data='/Users/yudaimatsuura/yolov8_env_train/datasets/data.yaml', epochs = 32, imgsz = 640)