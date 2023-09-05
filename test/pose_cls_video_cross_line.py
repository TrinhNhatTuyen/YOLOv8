import cv2, datetime, pyodbc, base64, requests, json
from ultralytics import YOLO
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from queue import Queue
import firebase_admin
from firebase_admin import credentials, messaging
from tensorflow.keras.models import load_model

humanpose_conf = 80
queue_len = 60
# cap = cv2.VideoCapture('rtsp://admin:Dat1qazxsw2@192.168.6.100/h264_stream')
video = "datatest/1.mp4"

def result_queue(q,value):
    if q.full():
        q.get()  # Lấy giá trị đầu tiên khi Queue đã đủ 60 phần tử
    q.put(value)  # Thêm giá trị mới vào Queue
    if q.queue.count(True) >= (queue_len-10):  # Kiểm tra nếu có ít nhất 50 True trong Queue
        if value:
            return True
        else:
            return False
    else:
        return False

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
    
def pose_cls_video(video, r_queue=False, save=False):
    # Load a model
    model = YOLO('YOLOv8n-pose.pt')
    pose_cls = load_model('pose_cls_v3.h5')
    thres = 80
    input = None
    
    x_bottom = 920
    x_top = 1020
    
    q_lockpicking = Queue(maxsize=queue_len)
    q_climbing = Queue(maxsize=queue_len)
    
    label_mapping = {
        0: 'none',
        1: 'lockpicking',
        2: 'climbing'
        }
#===========================================================================================================#  
    cap = cv2.VideoCapture(video)
    # Xác định thông số video đầu vào
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if save:
        # Tạo đối tượng VideoWriter để ghi video
        video_name = video.split('/')[-1].split('.')[0]
        out = cv2.VideoWriter('____output/'+video_name+'_pose_cls(70v3)_queue_cross_line.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
#===========================================================================================================#
    while cap.isOpened():
        ret, frame = cap.read()
        prob = 0
        if ret:

            # results = model(cv2.resize(frame, (640,640)), save=False)
            results = model(frame, save=False)   
            annotated_frame = results[0].plot(boxes=False)

            keypoints_arrs = results[0].keypoints.data.numpy()
            #===========================================================================================================#            # try:
            # if len(keypoints_arrs[0])==17:
            if len(results[0].boxes.data)>0:
                for skeleton in range(len(keypoints_arrs)):
                    # if float(results[0].boxes.data[0][4]*100)>humanpose_conf:
                    left_hand = np.round(keypoints_arrs[skeleton][9][:2]).astype(np.int32)
                    right_hand = np.round(keypoints_arrs[skeleton][10][:2]).astype(np.int32)
                    if float(results[0].boxes.data[skeleton][4]*100)>humanpose_conf and (point_position(x_top, x_bottom, left_hand)=='left' or point_position(x_top, x_bottom, right_hand)=='left'):
                        # input = keypoints_arrs[0,5:,:]
                        input = keypoints_arrs[skeleton,5:,:]
                        input[:,0] = input[:,0]/frame_width
                        input[:,1] = input[:,1]/frame_width
                        #===============================#
                        x_min = np.min(input[:,0])
                        y_min = np.min(input[:,1])
                        input[:,0] -= x_min
                        input[:,1] -= y_min
                        #===============================#    
                        input = np.expand_dims(input, axis=0)
                        pred = pose_cls.predict(input)[0]
                        max_id = np.argmax(pred)
                        label = label_mapping.get(max_id, 'unknown')
                        prob = np.max(pred)*100
                        #===========================================================================================================#
                        # Nếu phát hiện mở khóa
                        title="Có ăn trộm"
                        if prob >= thres and label=='lockpicking':
                            if r_queue:
                                if result_queue(q_lockpicking, True):
                                    text = 'Trộm bẻ khoá' + " {:.2f}%".format(prob)
                                    background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                                else:
                                    text = "{:.2f}%".format(prob)
                                    background_color = (0, 255, 0)      # Màu xanh (B, G, R)
                            else:
                                text = 'Trộm bẻ khoá' + " {:.2f}%".format(prob)
                                background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                        elif prob >= thres and label=='climbing':
                            if r_queue:
                                if result_queue(q_climbing, True):
                                    text = 'Trộm leo rào' + " {:.2f}%".format(prob)
                                    background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                                else:
                                    text = "{:.2f}%".format(prob)
                                    background_color = (0, 255, 0)      # Màu xanh (B, G, R)
                            else:
                                text = 'Trộm leo rào' + " {:.2f}%".format(prob)
                                background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                        else:
                            if r_queue:
                                result_queue(q_climbing, False)
                            text = 'Bình thường' + " {:.2f}%".format(prob)
                            background_color = (0, 255, 0)      # Màu xanh (B, G, R)
                        
                        #===========================================================================================================#                    
                        # Thiết lập font chữ
                        font_path = 'arial.ttf'
                        font_size = 40
                        font_color = (255, 255, 255)  # Màu trắng (B, G, R)
                        font = ImageFont.truetype(font_path, font_size)

                        
                        # Vẽ khung chữ nhật
                        left = int(results[0].boxes.data[skeleton][0])
                        top = int(results[0].boxes.data[skeleton][1])
                        right = int(results[0].boxes.data[skeleton][2])
                        bottom = int(results[0].boxes.data[skeleton][3])
                        cv2.rectangle(annotated_frame, (left,top), (right,bottom), background_color, thickness=3, lineType=cv2.LINE_AA)
                        
                        # Tạo một ảnh PIL từ hình ảnh Numpy
                        pil_image = Image.fromarray(annotated_frame)

                        # Tạo đối tượng vẽ trên ảnh PIL
                        draw = ImageDraw.Draw(pil_image)
                        
                        # Vẽ nền cho text 
                        text_width, text_height = draw.textsize(text, font=font)
                        left = int(results[0].boxes.data[skeleton][0])-2
                        top = int(results[0].boxes.data[skeleton][1])-text_height-3
                        rectangle_position = (left, top, left + text_width, top + text_height)
                        draw.rectangle(rectangle_position, fill=background_color)
                        
                        text_position = (left, top)
                        # Vẽ văn bản màu đỏ
                        draw.text(text_position, text, font=font, fill=font_color)
                        # Chuyển đổi ảnh PIL thành ảnh Numpy
                        annotated_frame = np.array(pil_image)
                    
#===========================================================================================================#            
            frame = cv2.resize(annotated_frame, 
                               (int((annotated_frame.shape[1])*0.6),int((annotated_frame.shape[0])*0.6)))
            if save:
                out.write(annotated_frame)
            cv2.imshow('YOLOv8 Inference', frame)
            
        else:
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    if save:
        out.release()
    cv2.destroyAllWindows()

# pose_cls_video('datatest/12_climb.mp4', save=True)
pose_cls_video('datatest/12_lockpicking2.mp4', save=True)
# pose_cls_video('datatest/theif.mp4', save=True)
# pose_cls_video('datatest/3.mp4', save=True)
# pose_cls_video('datatest/4.mp4', save=True)