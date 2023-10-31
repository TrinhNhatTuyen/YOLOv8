import cv2, os
from ultralytics import YOLO
detect_model = YOLO('yolov8n.pt')
seg_model = YOLO('yolov8n-seg')
frame = cv2.imread(os.path.join('C:/Users/Administrator/Documents/Zalo Received Files/testyolo/','12.jpg'))
scale=0.4
frame = cv2.resize(frame,(int((frame.shape[1])*(640/frame.shape[0])),int((frame.shape[0])*(640/frame.shape[0]))))
# results = seg_model(frame, save=True, hide_conf=True, conf=0.25, classes=[3], show=True)
# results[0].plot(boxes=False)
folder_path = 'C:/Users/Administrator/Documents/Zalo Received Files/14/'
r1 = detect_model(frame, save=False, show_conf=False, conf=0.3, show=False, classes=[3], show_labels=False)
r2 = seg_model(frame, save=False, show_conf=False, conf=0.3, show=False, classes=[3], show_labels=False)
cv2.imshow('Detection1', r1[0].plot())
cv2.imshow('Segment1', r2[0].plot())
print()

for i in os.listdir(folder_path):
    # pose_model = YOLO('yolov8n.pt')
    print(i)
    frame = cv2.imread(os.path.join(folder_path,i))
    # r1 = detect_model(frame, save=True, hide_conf=True, conf=0.2, show=False, classes=[0, 3])
    # r2 = seg_model(frame, save=True, hide_conf=True, conf=0.2, show=False, classes=[0, 3])
    # r1 = detect_model(frame, save=True, show_conf=False, conf=0.3, show=False, classes=[0], show_labels=False)
    # r2 = seg_model(frame, save=True, show_conf=False, conf=0.3, show=False, classes=[0], show_labels=False)
    r1 = detect_model(frame, save=True, show_conf=False, conf=0.3, show=False, classes=[3], show_labels=False)
    r2 = seg_model(frame, save=True, show_conf=False, conf=0.3, show=False, classes=[3], show_labels=False)
    cv2.imshow('Detection', r1[0].plot())
    cv2.imshow('Segment', r2[0].plot())
    print()
    