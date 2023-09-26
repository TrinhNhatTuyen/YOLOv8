import cv2
import pandas as pd
def detect_face(ori_img,detector):
    base_img = ori_img.copy()
    original_size = base_img.shape
    target_size = (300, 300)
    img = cv2.resize(ori_img, target_size)
    imageBlob = cv2.dnn.blobFromImage(image = img)
    detector.setInput(imageBlob)
    detections = detector.forward()
    column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
    detections_df = pd.DataFrame(detections[0][0], columns = column_labels)
    #0: background, 1: face
    detections_df = detections_df[detections_df['is_face'] == 1]
    detections_df = detections_df[detections_df['confidence'] >= 0.3]
    detections_df['left'] = (detections_df['left'] * 300).astype(int)
    detections_df['bottom'] = (detections_df['bottom'] * 300).astype(int)
    detections_df['right'] = (detections_df['right'] * 300).astype(int)
    detections_df['top'] = (detections_df['top'] * 300).astype(int)
    return detections_df