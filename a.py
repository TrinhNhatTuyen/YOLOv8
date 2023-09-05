import cv2, pickle
from ultralytics import YOLO
import numpy as np

# Load a model
model = YOLO('YOLOv8n-pose.pt')
# model = YOLO('YOLOv8s-pose.pt')
# model = YOLO('YOLOv8m-pose.pt')
# model = YOLO('YOLOv8l-pose.pt')
# model = YOLO('YOLOv8x-pose.pt')
# model = YOLO('yolov8x-pose-p6.pt')

# cap = cv2.VideoCapture('rtsp://admin:Dat1qazxsw2@192.168.6.100/h264_stream')
# cap = cv2.VideoCapture('D:\Code\Pose\MediaPipe_MiAI\Trộm trèo tường vào nhà trộm cây cảnh _ Vuhoangtelecom1.mp4')
cap = cv2.VideoCapture('D:/Code/datatest/Trim.mp4')

############ ------------>>> https://docs.ultralytics.com/modes/predict/#arguments <<<------------ ############

while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:

        # results = model(cv2.resize(frame, (640,640)), save=False)
        results = model(frame, save=False)
        
        annotated_frame = results[0].plot()
# ------------------------------------------------------------------------ #
        """
            Lấy thông tin keypoint:
            Mỗi keypoints có 3 giá trị (x, y, confidence), trong đó:
            * x là tọa độ x (hoành độ) của keypoints.
            * y là tọa độ y (tung độ) của keypoints.
            * confidence là độ tin cậy (confidence) của keypoints.
        """
        
        keypoints_arrs = results[0].keypoints.data.numpy()
# ------------------------------------------------------------------------ #
        
        cv2.imshow('YOLOv8 Inference', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows()
