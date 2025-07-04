import cv2
from ultralytics import YOLO

model = YOLO("/home/go2laptop/yudai_ws/src/semantic_segmentation_terrain/scripts/runs/detect/train4/weights/best.pt")

video_path = "/home/go2laptop/yudai_ws/input_video2.mp4"

output_path = "/home/go2laptop/yudai_ws/output_video.mp4"
cap  = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame ,conf = 0.05, iou = 0.3)

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    cv2.imshow("YOLOv8 Inference", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()