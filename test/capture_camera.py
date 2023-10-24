import cv2
import requests
import numpy as np

# Địa chỉ RTSP của camera
rtsp_url = 'rtsp://admin:Vinaai!123@py1ai.cameraddns.net:5543/cam/realmonitor?channel=1&subtype=0&unicast=true'

# Tạo một đối tượng VideoCapture để đọc video từ URL RTSP
cap = cv2.VideoCapture(rtsp_url)

# Kiểm tra xem camera có sẵn sàng không
if not cap.isOpened():
    print("Không thể kết nối đến camera.")
    exit()

# Đọc một khung hình từ video stream
for i in range(10):
    ret, frame = cap.read()

# Kiểm tra xem việc đọc khung hình có thành công không
if not ret:
    print("Không thể đọc khung hình từ camera.")
    cap.release()
    exit()

# Resize ảnh về kích thước 1920x1080
# frame = cv2.resize(frame, (1920, 1080))

# Lưu ảnh đã resize vào thư mục "test"
cv2.imwrite('test/Cam 4_noresize.jpg', frame)

# Đóng camera
cap.release()
