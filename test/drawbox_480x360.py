import cv2

# Đọc ảnh từ đường dẫn
image_path = 'test/Cam 4_noresize.jpg'
image = cv2.imread(image_path)
base_img = image.copy()
# Danh sách các tọa độ pixel (x, y) ở góc trái trên của các khung
coordinates = [(1850, 130), (1850, 130+360), (1850-480, 130+180), (1850-480, 130+180+360)]

frame_color = (0, 0, 255)  # Màu đỏ
check_face = []

frame_width = 480
frame_height = 360
scale_percent = 0.5
# Vẽ khung màu đỏ tại các tọa độ pixel
for (x, y) in coordinates:
    cv2.rectangle(image, (x, y), (x + frame_width, y + frame_height), frame_color, thickness=2)
    roi = base_img[y:y + frame_height, x:x + frame_width]
    check_face.append(roi)
# Hiển thị hình kết quả
while True:
    cv2.imshow('Result Image', cv2.resize(image, None, fx=scale_percent, fy=scale_percent))
    for idx, box in enumerate(check_face):
        cv2.imshow(f'Box {idx + 1}', box)
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

# Đóng cửa sổ khi người dùng bấm q
cv2.destroyAllWindows()
