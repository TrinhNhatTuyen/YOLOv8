import cv2, time, os, sys
import pandas as pd
from detect_face_img import *

coordinates = [(1850, 130), (1850, 130+360), (1850-480, 130+180), (1850-480, 130+180+360)]
frame_color = (0, 0, 255)  # Màu đỏ
width = 480
height = 360
scale_percent = 0.5
video_name = 'test/vlc-record-2023-10-04-15h32m10s-rtsp___py1ai.cameraddns.net_5543_cam_realmonitor-.mp4'
cap = cv2.VideoCapture(video_name)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(video_name.split('.mp4')[0]+'_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
# Kiểm tra xem có thể mở webcam hay không
if not cap.isOpened():
    print("Không thể mở webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc khung hình từ webcam.")
        break

    base_img = frame.copy()   
    check_face = []

    for (x, y) in coordinates:
        cv2.rectangle(frame, (x, y), (x + width, y + height), frame_color, thickness=2)
        # roi = base_img[y:y + frame_height, x:x + frame_width]
        frame[y:y + height, x:x + width] = detect_face(ori_img=frame[y:y + height, x:x + width], detector=detector)
        # check_face.append(roi)

    out.write(frame)
    cv2.imshow('Webcam', cv2.resize(frame, None, fx=scale_percent, fy=scale_percent))

    # Đợi 1ms và kiểm tra xem người dùng có ấn phím 'q' không để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        out.release()
        break

# Giải phóng webcam và đóng các cửa sổ hiển thị
out.release()
cap.release()
cv2.destroyAllWindows()