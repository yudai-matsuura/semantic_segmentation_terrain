from ultralytics import YOLO

if __name__ == "__main__":
    mode1 = YOLO('yolov8n.pt')
    mode1.train(data='/home/go2laptop/yudai_ws/YOLO_terrain.v1i.yolov8/data.yaml', epochs = 32, imgsz = 640)