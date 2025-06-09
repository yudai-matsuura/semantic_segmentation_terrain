import cv2
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.abspath(os.path.join(script_dir, '..', 'data', 'train_video', 'input_video.mp4'))
output_image_dir = os.path.abspath(os.path.join(script_dir, '..', 'data', 'train_images'))

os.makedirs(output_image_dir, exist_ok=True)

cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print("Could not open video file")
    exit()

frame_count = 0
display_width = 1024

print("Press SPACE to save frame, Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video has finished")
        break

    height, width = frame.shape[:2]

    resize_ratio = display_width / width
    new_height = int(height * resize_ratio)
    resized_frame = cv2.resize(frame,(display_width, new_height))

    cv2.imshow("Video Frame", resized_frame)

    key = cv2.waitKey(1) & 0xFF

    # if space key pressed
    if key == ord(' '):
        frame_filename = os.path.join(output_image_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        print(f"saved frame: {frame_filename}")
    # press q to quit
    if key == ord("q"):
        print("quitting")
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()