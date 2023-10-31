import cv2

# Đường dẫn đến tệp video
video_path = r'D:/Code/Pose/YOLOv8/smthgelse2/lockpicking_5(2).mp4'

try:
    # Tạo một đối tượng VideoCapture
    cap = cv2.VideoCapture(video_path)

    # Kiểm tra xem việc mở video có thành công không
    if not cap.isOpened():
        print("Không thể mở video.")
    else:
        # Lặp qua từng khung hình trong video
        while True:
            # Đọc một khung hình
            ret, frame = cap.read()

            # Kiểm tra xem liệu đã đọc đến cuối video chưa
            if not ret:
                break

            # Hiển thị khung hình
            cv2.imshow('Video', frame)

            # Đợi một chút để hiển thị khung hình, thời gian đợi có thể điều chỉnh
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Giải phóng tài nguyên và đóng cửa sổ khi kết thúc
        cap.release()
        cv2.destroyAllWindows()

except Exception as e:
    print(f"Lỗi: {str(e)}")
