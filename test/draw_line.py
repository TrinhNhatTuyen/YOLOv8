import cv2
import numpy as np
# from ultralytics import YOLO
# model = YOLO('yolov8n-pose.pt')


def scale_img(img, scale):
    height, width, _ = img.shape
    scaled_image = cv2.resize(img, (width * scale // 100, height * scale // 100))
    return scaled_image

def point_position(x_top, x_bottom, point_test, img_w=1920, img_h=1080):

    p1_Oxy = np.array([x_bottom, 0])
    p2_Oxy = np.array([x_top, img_h])
    
    # Chuyển tung độ của điểm cần check từ hệ Oxy của ảnh sang hệ Oxy thông thường
    p_test_Oxy = np.array([point_test[0], img_h - point_test[1]])
    # line_vector = p2 - p1
    # test_vector = p_test - p1
    line_vector = p1_Oxy - p2_Oxy
    test_vector = p_test_Oxy - p2_Oxy
    cross_product = np.cross(line_vector, test_vector)
    if cross_product > 0:
        return "right"
    elif cross_product < 0:
        return "left"
    else:
        return "on"

x_bottom = 920
x_top = 1020
point_a = np.array([x_bottom, 0])
point_b = np.array([x_top, 1080])

# print(point_position(x_top=1020, x_bottom=920, point_test=np.array([919, 1080])))
# print(point_position(x_top=1020, x_bottom=920, point_test=np.array([920, 1080])))
# print(point_position(x_top=1020, x_bottom=920, point_test=np.array([921, 1080])))

# print(point_position(x_top=1020, x_bottom=920, point_test=np.array([1019, 0])))
# print(point_position(x_top=1020, x_bottom=920, point_test=np.array([1020, 0])))
# print(point_position(x_top=1020, x_bottom=920, point_test=np.array([1021, 0])))

# print(point_position(x_top=1020, x_bottom=920, point_test=np.array([973, 224])))
# print(point_position(x_top=1020, x_bottom=920, point_test=np.array([969, 215])))


def draw_line_on_image_and_save():
    image = cv2.imread("test/True.png")

    if image is None:
        print("Không thể đọc ảnh.")
        return

    point_a = np.array([x_bottom, 1080])
    point_b = np.array([x_top, 0])

    # results = model(image, save=False)
    # keypoints_arrs = results[0].keypoints.data.numpy()
    # left_hand = np.round(keypoints_arrs[0][9][:2]).astype(np.int32)
    # right_hand = np.round(keypoints_arrs[0][10][:2]).astype(np.int32)
    # annotated_frame = results[0].plot(boxes=False, show_conf=False)
    
    annotated_frame = image
    annotated_frame = cv2.line(annotated_frame, point_a, point_b, (0, 0, 255), 2)  # Tạo bản sao để vẽ đường
    annotated_frame = cv2.rectangle(annotated_frame, (905, 180), (990, 250), (0, 0, 255), 2)
    # annotated_frame = cv2.circle(annotated_frame, left_hand, 5, (255, 0, 0), -1)
    # annotated_frame = cv2.circle(annotated_frame, right_hand, 5, (255, 0, 0), -1)
    cv2.imwrite("test/output_image.png", annotated_frame)
    scaled_image = scale_img(annotated_frame,60)
    
    
    
    # Hiển thị ảnh và chờ phím "q" được bấm
    # while point_position(x_top, x_bottom, left_hand)=='left' or point_position(x_top, x_bottom, right_hand)=='left':
    while True:
        cv2.imshow("Image with Line", scaled_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

draw_line_on_image_and_save()
