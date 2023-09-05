import cv2
from ultralytics import YOLO

# Load a model
model = YOLO('YOLOv8n.pt')

# cap = cv2.VideoCapture('rtsp://admin:Dat1qazxsw2@192.168.6.100/h264_stream')
video = 'D:/Code/datatest/dao.mp4'
cap = cv2.VideoCapture(video)

# Xác định thông số video đầu vào
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Tạo đối tượng VideoWriter để ghi video
# out = cv2.VideoWriter(video.split('.')[0]+'_pose.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if ret:

        # results = model(cv2.resize(frame, (640,640)), save=False)
        results = model(frame, save=False)
        
        annotated_frame = results[0].plot()
        if 43 in results[0].boxes.cls.int().tolist():
            print("Knife")
        # out.write(annotated_frame)
        
        # cv2.putText(annotated_frame, "fps: {:.2f}".format(fps), (20,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 155, 255), 2)
        frame = cv2.resize(annotated_frame, ((int)((annotated_frame.shape[1])*0.6),(int)((annotated_frame.shape[0])*0.6)))
        cv2.imshow('YOLOv8 Inference', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
cap.release()
# out.release()
cv2.destroyAllWindows()