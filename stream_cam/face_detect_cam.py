import cv2, time, os, sys
import pandas as pd
sys.path.append('D:\Code\Pose\YOLOv8')
from overstepframe import FreshestFrame

# Tạo đối tượng detector từ mô hình Caffe
prototxt_path = "stream_cam/pre_model/deploy.prototxt"
caffemodel_path = "stream_cam/pre_model/res10_300x300_ssd_iter_140000.caffemodel"
detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

scale_percent = 0.3

def detect_face(ori_img, detector):
    base_img = ori_img.copy()
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
    
    # Vẽ hình chữ nhật các khuôn mặt
    for _, row in detections_df.iterrows():
        left = row['left']
        top = row['top']
        right = row['right']
        bottom = row['bottom']
        cv2.rectangle(base_img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
        
        cv2.imshow('Face Detected', cv2.resize(base_img, None, fx=scale_percent, fy=scale_percent))
    return base_img

# url = 'rtsp://admin:Dat1qazxsw2@192.168.6.100:1554/h264_stream'
# url = 'rtsp://admin:1qazxsw2@vinaai.ddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true' # Link stream video RTSP
# url = 'rtsp://admin:NuQuynhAnh@cam24423linhdong.smartddns.tv:1554/cam/realmonitor?channel=1&subtype=0&unicast=true'

# url = 'rtsp://admin:NuQuynhAnh@cam14423linhdong.smartddns.tv:1554/cam/realmonitor?channel=1&subtype=0&unicast=true' # Cam 1
# url = 'rtsp://admin:Admin123@mtkhp2408.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true'         # Cam 2
# url = 'rtsp://admin:Admin123@mtkhp2420.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true'         # Cam 3

url = [
        # 'rtsp://admin:Vinaai!123@py1ai.cameraddns.net:5543/cam/realmonitor?channel=1&subtype=0&unicast=true',           # Cam 4
        # 'rtsp://admin:Vinaai!123@py1ai.cameraddns.net:5541/cam/realmonitor?channel=1&subtype=0&unicast=true',           # Cam 5
        'rtsp://admin:Vinaai!123@py2ai.cameraddns.net:5541/cam/realmonitor?channel=1&subtype=0&unicast=true',           # Cam 6
        'rtsp://admin:Vinaai!123@py2ai.cameraddns.net:5543/cam/realmonitor?channel=1&subtype=0&unicast=true',           # Cam 7
        'rtsp://admin:Vinaai!123@py2ai.cameraddns.net:5545/cam/realmonitor?channel=1&subtype=0&unicast=true',           # Cam 8
        ]

fresh, frame, detected_frame, cnt, first_frame, second_frame, None_frame, cam_name = [], [], [], [], [], [], [], []
for i in range(len(url)):
    fresh.append(object())
    frame.append(object())
    detected_frame.append(object())
    cnt.append(0)
    first_frame.append(None)
    second_frame.append(None)
    None_frame.append(0)
    cam_name.append(f'Cam {i+4}')
    
    fresh[i]=FreshestFrame(cv2.VideoCapture(url[i]))
    
# fresh = object()
# fresh = FreshestFrame(cv2.VideoCapture(url))
# frame = object()
# cnt = 0

try:
    while True:
        for CC in range(len(url)):
            cnt[CC],frame[CC] = fresh[CC].read(seqnumber=cnt[CC]+1, timeout=5)
            if not cnt[CC]:
                print(f"Timeout, can't read new frame of cam {CC}!")
                raise Exception()
            
            if first_frame[CC] is None:
                first_frame[CC] = frame[CC]
                None_frame[CC]+=1
                continue
            
            detected_frame[CC] = detect_face(frame[CC], detector)
            cv2.imshow(cam_name[CC], cv2.resize(detected_frame[CC], None, fx=scale_percent, fy=scale_percent))
            print(CC+1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise Exception("Stop")
except:
    pass

#-------------------------------------------------------------------------------------------------------------

