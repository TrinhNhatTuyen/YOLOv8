import cv2
import numpy as np

# Đọc ảnh từ tệp datatest/camtrong.jpg
image_path = 'datatest/camtrong.jpg'
image = cv2.imread(image_path)

# Tọa độ của các điểm A, B, C, D cho các hình chữ nhật L0 và L1
rectangles = {
    "L0": {"A": [650, 300], "B": [757, 300], "C": [757, 432], "D": [650, 432]},
    "L1": {"A": [1240, 330], "B": [1340, 330], "C": [1340, 445], "D": [1240, 445]}
}

# Vẽ hình chữ nhật màu đỏ trên ảnh
for rect_name, rect_coords in rectangles.items():
    pts = np.array([rect_coords["A"], rect_coords["B"], rect_coords["C"], rect_coords["D"]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

# Thay đổi kích thước ảnh về 60%
height, width = image.shape[:2]
resized_image = cv2.resize(image, (int(width * 0.6), int(height * 0.6)))

# Hiển thị ảnh đã thay đổi kích thước
cv2.imshow('Anh da thay doi kich thuoc', resized_image)

# Chờ người dùng nhấn phím bất kỳ để thoát
cv2.waitKey(0)
cv2.destroyAllWindows()
