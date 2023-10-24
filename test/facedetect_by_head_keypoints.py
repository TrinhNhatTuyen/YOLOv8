import cv2, time, os, sys
import pandas as pd
import numpy as np
from detect_face_img import *
from ultralytics import YOLO
model = YOLO('YOLOv8n-pose.pt')
# results = model('test/i.jpg', save=True)  # results list
# from keras.models import load_model
# pose_cls = load_model('pose_cls_v3.h5')
#---------------------------------------------------------------------
frame_color = (0, 0, 255)  # Màu đỏ
half_width = 1080
half_height = 720
w_box4detectface = 480
h_box4detectface = 360
humanpose_conf = 50
scale_percent = 0.5
#---------------------------------------------------------------------
video_name = 'test/vlc_record_cam4_(1).mp4'
cap = cv2.VideoCapture(video_name)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
half_width = int(frame_width/2)
half_height = int(frame_height/2)
#---------------------------------------------------------------------
w = frame_width-1
h = frame_height-1
# coordinates = [
#                 (w-half_width, h-half_height), 
#                 (w-half_width, h-2*half_height), 
#                 (0, h-half_height), 
#                 (w-2*half_width, h-2*half_height),
#             ]
coordinates = [
                # (0, 0), 
                (half_width, 0), 
                (0, half_height), 
                (half_width, half_height),
            ]
#---------------------------------------------------------------------
image = cv2.imread('test/7.jpg')
annotated_frame = image.copy()
# Vẽ khung màu đỏ tại các tọa độ pixel
for (x, y) in coordinates:
    # cv2.rectangle(image, (x, y), (x + half_width, y + half_height), frame_color, thickness=2)
    
    # subframes = [
    #                 image[:half_height, :half_width],   # Top-left corner (góc trên bên trái)
    #                 image[:half_height, half_width:],   # Top-right corner (góc trên bên phải)
    #                 image[half_height:, :half_width],   # Bottom-left corner (góc dưới bên trái)
    #                 image[half_height:, half_width:],   # Bottom-right corner (góc dưới bên phải)
    #             ]
    
    # 1/4 image
    subframe = image[y:y+half_height, x:x+half_width]
    results = model(subframe, save=False)
    keypoints_arrs = results[0].keypoints.data.numpy()
    while True:
        cv2.imshow(f'1/4 frame, x={x} y={y}', results[0].plot(boxes=False))
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
    if len(results[0].boxes.data)>0:
        for skeleton in range(len(keypoints_arrs)):
            if float(results[0].boxes.data[skeleton][4]*100)>humanpose_conf:
                
                #--------------------------------------Face--------------------------------------
                current_skeleton = keypoints_arrs[skeleton]
                highestVisible = np.argmax(current_skeleton[:5,2])
                # x1, y1 là tọa độ 1 trong 5 điểm trên đầu có Visible cao nhất
                x1 = int(current_skeleton[highestVisible,0] - w_box4detectface/2)
                y1 = int(current_skeleton[highestVisible,1] - h_box4detectface/2)
                # Chuyển về tọa độ trên ảnh gốc "image"
                x1 = x1 + x
                y1 = y1 + y
                
                # Trường hợp topleft nằm ngoài frame
                if x1<0: x1=0
                if y1<0: y1=0
                if x1>frame_width-w_box4detectface: x1=frame_width-w_box4detectface
                if y1>frame_height-h_box4detectface: y1=frame_height-h_box4detectface
                
                
                # Detect Face
                box4detectface = image[y1:y1 + h_box4detectface, x1:x1 + w_box4detectface].copy()
                # Các tọa độ của face: top, left, right, bottom trong box 480x360
                top_f, left_f, right_f, bottom_f = detect_face(ori_img=box4detectface, 
                                                    detector=detector,
                                                    drawbox_on=annotated_frame[y1:y1 + h_box4detectface, x1:x1 + w_box4detectface])
                
                if top_f is not None and left_f is not None and right_f is not None and bottom_f is not None:
                    # Chuyển về tọa độ trên ảnh gốc "image"
                    top_f = top_f + y1
                    left_f = left_f + x1
                    right_f = right_f + x1
                    bottom_f = bottom_f + y1
                
                #--------------------------------------Skeleton--------------------------------------
                # Tọa độ của box skeleton
                left_sklt_box1 = int(results[0].boxes.data[skeleton][0])
                top_sklt_box1 = int(results[0].boxes.data[skeleton][1])
                right_sklt_box1 = int(results[0].boxes.data[skeleton][2])
                bottom_sklt_box1 = int(results[0].boxes.data[skeleton][3])
                
                # Crop lại ảnh có tỷ lệ gần 1920x1080 (1.77)
                h_box4detectskeleton = int(1.8*(bottom_sklt_box1-top_sklt_box1))
                w_box4detectskeleton = int(1.33*h_box4detectskeleton)
                
                top_detectsklt2 = int((bottom_sklt_box1+top_sklt_box1)/2 - h_box4detectskeleton/2)
                left_detectsklt2 = int((left_sklt_box1+right_sklt_box1)/2 - w_box4detectskeleton/2)
                
                # Chuyển về tọa độ trên ảnh gốc "image"
                top_detectsklt2 = top_detectsklt2 + y
                left_detectsklt2 = left_detectsklt2 + x
                print(left_detectsklt2, top_detectsklt2)
                
                # Trường hợp topleft nằm ngoài frame
                if top_detectsklt2<0: 
                    top_detectsklt2=0
                if left_detectsklt2<0: 
                    left_detectsklt2=0
                if top_detectsklt2>frame_height-h_box4detectskeleton: 
                    top_detectsklt2=frame_height-h_box4detectskeleton
                if left_detectsklt2>frame_width-w_box4detectskeleton: 
                    left_detectsklt2=frame_width-w_box4detectskeleton
                
                # Detect khung xương lần 2
                subframe2 = image[top_detectsklt2:top_detectsklt2 + h_box4detectskeleton, 
                                    left_detectsklt2:left_detectsklt2 + w_box4detectskeleton].copy()
                results2 = model(subframe2, save=False)
                keypoints_arrs2 = results2[0].keypoints.data.numpy()
                while True:
                    cv2.imshow(f'x={x} y={y}', results2[0].plot(boxes=False))
                    key = cv2.waitKey(0)
                    if key == ord('q'):
                        cv2.destroyAllWindows()
                        break
                firstsklt = 0
                if len(results2[0].boxes.data)==0:
                    pass
                elif float(results[0].boxes.data[firstsklt][4]*100)<humanpose_conf:
                    pass
                else:
                    # Tọa độ của box skeleton lần 2, chỉ lấy 1 khung xương đầu tiên
                    left_sklt_box2 = int(results2[0].boxes.data[firstsklt][0])
                    top_sklt_box2 = int(results2[0].boxes.data[firstsklt][1])
                    right_sklt_box2 = int(results2[0].boxes.data[firstsklt][2])
                    bottom_sklt_box2 = int(results2[0].boxes.data[firstsklt][3])
                    crop_skl_box = results2[0].plot(boxes=False)[top_sklt_box2:bottom_sklt_box2,
                                                                left_sklt_box2:right_sklt_box2]
                    
                    # Chuyển về tọa độ trên ảnh gốc "image"
                    w_sklt_box2 = right_sklt_box2 - left_sklt_box2
                    h_sklt_box2 = bottom_sklt_box2 - top_sklt_box2
                    
                    top_sklt_box2 = top_sklt_box2 + top_detectsklt2
                    left_sklt_box2 = left_sklt_box2 + left_detectsklt2

                    # Trường hợp nằm ngoài frame
                    # if top_sklt_box2<0: top_sklt_box2=0
                    # if left_sklt_box2<0: left_sklt_box2=0
                    # if top_sklt_box2>frame_width-w_sklt_box2: top_sklt_box2=frame_width-w_sklt_box2
                    # if left_sklt_box2>frame_height-h_sklt_box2: left_sklt_box2=frame_height-h_sklt_box2
                    
                    # Gắn box detect được sau cùng vào annotated_frame 
                    annotated_frame[top_sklt_box2:top_sklt_box2+h_sklt_box2,
                                    left_sklt_box2:left_sklt_box2+w_sklt_box2] = crop_skl_box

                    print()
                    
                    
    # annotated_frame = results[0].plot(boxes=False)
    # image[y:y + half_height, x:x + half_width] = detect_face(ori_img=image[y:y + half_height, x:x + half_width], detector=detector)

# Hiển thị hình kết quả
while True:
    cv2.imshow('Result Image', cv2.resize(annotated_frame, None, fx=scale_percent, fy=scale_percent))
    key = cv2.waitKey(0)
    if key == ord('q'):
        break

# Đóng cửa sổ khi người dùng bấm q
cv2.destroyAllWindows()
