import cv2, time, os, sys
import pandas as pd

prototxt_path = "stream_cam/pre_model/deploy.prototxt"
caffemodel_path = "stream_cam/pre_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
# scale_percent = 0.7

def detect_face(ori_img, detector, drawbox_on=None):
    base_img = ori_img.copy()
    #-------------------------------------
    
    # top_left = (1470, 55)
    # bottom_right = (2500, 600)
    # base_img = ori_img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    #-------------------------------------
    original_size = base_img.shape
    target_size = (300, 300)
    img = cv2.resize(ori_img, target_size)
    # imageBlob = cv2.dnn.blobFromImage(image=img)
    imageBlob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    detector.setInput(imageBlob)
    detections = detector.forward()
    column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
    detections_df = pd.DataFrame(detections[0][0], columns=column_labels)
    detections_df = detections_df[detections_df['is_face'] == 1]
    detections_df = detections_df[detections_df['confidence'] >= 0.3]
    detections_df['left'] = (detections_df['left'] * original_size[1]).astype(int)
    detections_df['bottom'] = (detections_df['bottom'] * original_size[0]).astype(int)
    detections_df['right'] = (detections_df['right'] * original_size[1]).astype(int)
    detections_df['top'] = (detections_df['top'] * original_size[0]).astype(int)
    
    left = None
    top = None
    right = None
    bottom = None
    # Vẽ hình chữ nhật các khuôn mặt
    for _, row in detections_df.iterrows():
        left = row['left']
        top = row['top']
        right = row['right']
        bottom = row['bottom']
        if drawbox_on is None:
            cv2.rectangle(base_img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
        else:
            cv2.rectangle(drawbox_on, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
        
    # while True:
    #     cv2.imshow('Face Detected', base_img)
    #     # cv2.imshow('Face Detected', cv2.resize(base_img, None, fx=scale_percent, fy=scale_percent))
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q'):
    #         break
    
    # if drawbox_on is None:
    #     return top, left, right, bottom
    # else:
    return top, left, right, bottom

# ori_img = cv2.imread('test/Cam 4_noresize - Face2.jpg')
# ori_img = cv2.imread('D:/Code/Pose/pytorch-openpose-master/images/dance1.jpg')
# ori_img = cv2.imread('test/streamCam4_1_480x360.jpg')
# detect_face(ori_img, detector)