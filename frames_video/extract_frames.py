import cv2
import os

video_path = "C:\\Users\\RaghavBansal\\OneDrive - ProcDNA Analytics Pvt. Ltd\\Desktop\\amass\\PoseDataParticleFilter\\PoseDataParticleFilter\\frames_video\\912_3_01.mp4"
output_folder = "frames"

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