import cv2
import os

video_path = r"frames_video\Subject_1_F_1.mp4"
output_folder = r"frames\bml_movi_renders"

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_path = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
    cv2.imwrite(frame_path, frame)

    frame_count += 1

cap.release()

print(f"Extracted {frame_count} frames.")