from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
model = YOLO('yolov8n-pose.pt')  # load an official model

source = 'D:/Code/datatest/keypoint_index.jpg'
source = 'D:/Code/datatest/Kinh ngạc cảnh kẻ trộm phá két sắt cửa hàng FPT trộm tiền tỉ.mp4'
# source = 'D:/Code/datatest/mask(2).jpg'
model.predict(source, save=True, imgsz = 320, conf = 0.5, show_conf = False)
# for r in result:
#     data = r.boxes.data  # Boxes object for bbox outputs
#     orig_img = r.orig_img
#     for i in data:
#         cv2.rectangle(orig_img,
#                       (int(i[0]),int(i[1])),
#                       (int(i[2]),int(i[3])),
#                       (0, 0, 255))

# print()