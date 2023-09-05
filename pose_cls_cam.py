import cv2, requests, base64, json, datetime
from ultralytics import YOLO
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from queue import Queue
from tensorflow.keras.models import load_model
from overstepframe import FreshestFrame
# from firebase_admin import credentials, messaging


humanpose_conf = 90
queue_len = 60
# ip = '192.168.6.17'
ip = '125.253.117.120'
# fcm = 'dSZbYbanSl-pIr8eBcL2KN:APA91bHsX7uv4J2TdaoEbxsZg9y3U_Y54QWkkBCw8Xko8It0-w5XbFY5ae6VIiM1iT_r-xDyzF_gq0jCorYx5aBN7OL49ULuC9ay5n1dUmCKO0X3HYa5Dv3X8aV7faym47ZcJWPBYhwo'
# fcm = 'dwiwc-hqSSqauJ8sXPQkCM:APA91bH0kZuNqUxm1Q-vOuHgHvpfiEAA6gmkbAenmG_6pbFwZ-0QwrGg03sTaVBfKvySGSRk2w24mM4zGRNPjxpNHT9wXgywsDZGjumWPyYoxr-LzQ6PoIqb0Bl9HTOFIC522SDkK8f6'
# fcm = 'fNM-TsWQSqmLsBUg9HejwR:APA91bG8owUPOWHA0mCHs4f8Pi3Pqtus0iLszlPajoeX2nQtYkQ8v6LpDe3n8b1zDI2FLxUKOs_fosMrkc-7TA_bN2kY9B8GGd1xe89GQESaL6Ir5Qlz3-zA2uFEe4Xd-KB55PEtxS32'
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

def get_camera_id(rtsp):
    api_url = 'http://'+ip+':5000/api/notification/get-camera-id'
    data = {
        'key': '5c1f45bde9d2aff92e03acbac0b6d49f6410ca490c1fe85a082650ee9c23f63d',
        'rtsp': rtsp,

    }
    response = requests.post(api_url, json=data)
    return response.json()['message']

def save_notification(frame, camera_id, title, body):
    api_url = 'http://'+ip+':5000/api/notification/save'
    # Chuyển đổi dữ liệu ảnh thành chuỗi base64
    _, image_data = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(image_data).decode("utf-8")
    
    data = {
        'key': '5c1f45bde9d2aff92e03acbac0b6d49f6410ca490c1fe85a082650ee9c23f63d',
        'camera_id': camera_id,
        'notification_type': 'Alert',
        'title': title,
        'body': body,
        'base64': base64_image,
    }
    response = requests.post(api_url, json=data)
    print(response.json())
    
def get_fcm_to_send(camera_id):
    api_url = 'http://'+ip+':5000/api/notification/get-fcm-to-send'
    data = {
        'key': '5c1f45bde9d2aff92e03acbac0b6d49f6410ca490c1fe85a082650ee9c23f63d',
        'camera_id': camera_id,
    }
    response = requests.post(api_url, json=data)
    return json.loads(response.text)

def pose_fcm(fcms, title, body, data=None):
    for fcm in fcms:
        # Đường dẫn API FCM
        url = 'https://fcm.googleapis.com/fcm/send'
        
        # Đặt thông báo đẩy
        payload = {
            'to': fcm,
            'notification': {
                'title': title,
                'body': body
            },
            'image': 'https://cdn.pixabay.com/photo/2017/09/01/00/15/png-2702691_640.png'  
        }
        
        # Thêm dữ liệu tùy chỉnh (nếu có)
        if data:
            payload['data'] = data
        
        # Đặt tiêu đề của thông báo gửi tới FCM
        headers = {
            'Authorization': 'Key=AAAAUM0_kA0:APA91bFq6fvEmRIHZrF4VYTpTcsZHDo_bXvfm1jearG3A8BuNh_pEHtQtYhfGkbDkzsPm_lEwSh-t1LKB50c89wTaEs6N_RAqw7-JhNoUgmA_S5XyNA63E9MICw19QGwCSshw_o_sefG',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(url, json=payload, headers=headers)
        if response.status_code == 200:
            print('Thông báo đẩy đã được gửi thành công.')
        else:
            print('Gửi thông báo đẩy không thành công. Mã lỗi:', response.status_code)
    
def pose_cls_video(url, r_queue=False):
    # fcm = 'dSZbYbanSl-pIr8eBcL2KN:APA91bHsX7uv4J2TdaoEbxsZg9y3U_Y54QWkkBCw8Xko8It0-w5XbFY5ae6VIiM1iT_r-xDyzF_gq0jCorYx5aBN7OL49ULuC9ay5n1dUmCKO0X3HYa5Dv3X8aV7faym47ZcJWPBYhwo'
    # fcm = 'dwiwc-hqSSqauJ8sXPQkCM:APA91bH0kZuNqUxm1Q-vOuHgHvpfiEAA6gmkbAenmG_6pbFwZ-0QwrGg03sTaVBfKvySGSRk2w24mM4zGRNPjxpNHT9wXgywsDZGjumWPyYoxr-LzQ6PoIqb0Bl9HTOFIC522SDkK8f6'
    # fcm = 'fNM-TsWQSqmLsBUg9HejwR:APA91bG8owUPOWHA0mCHs4f8Pi3Pqtus0iLszlPajoeX2nQtYkQ8v6LpDe3n8b1zDI2FLxUKOs_fosMrkc-7TA_bN2kY9B8GGd1xe89GQESaL6Ir5Qlz3-zA2uFEe4Xd-KB55PEtxS32'
    camera_id = get_camera_id(url)
    list_fcm = get_fcm_to_send(camera_id)
    fresh = object()
    fresh = FreshestFrame(cv2.VideoCapture(url))
    frame = object()
    cnt = 0
    scale = 0.7
    data = {
        'key1': 'value1',
        'key2': 'value2'
    }
    # Load a model
    model = YOLO('YOLOv8n-pose.pt')
    pose_cls = load_model('pose_cls_v3.h5')
    thres = 80
    input = None
    
    q_lockpicking = Queue(maxsize=queue_len)
    q_climbing = Queue(maxsize=queue_len)
    
    label_mapping = {
        0: 'none',
        1: 'lockpicking',
        2: 'climbing'
        }

#===========================================================================================================#
    try:
        while True:
            t = datetime.datetime.now()
            if (t.minute % 5 == 0) and t.second<2:
                raise Exception("Restarting")
            
            cnt, frame = fresh.read(seqnumber=cnt+1)
            try:
                frame_width = frame.shape[1]
            except:
                continue
            prob = 0
            # try:

            # results = model(cv2.resize(frame, (640,640)), save=False)
            results = model(frame, save=False)
            annotated_frame = results[0].plot(boxes=False)
            
            keypoints_arrs = results[0].keypoints.data.numpy()
    #===========================================================================================================#            # try:
            if len(keypoints_arrs[0])==17:
            # for i in range(len(keypoints_arrs)):
                if float(results[0].boxes.data[0][4]*100)>humanpose_conf:
                # if results[0].boxes.data[i][4]>80:
                    input = keypoints_arrs[0,5:,:]
                    # input = keypoints_arrs[i,5:,:]
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
                                # text = "{:.2f}%".format(prob)
                                save_notification(frame, camera_id, title=title, body=text) # Lưu lại thông tin cảnh báo
                                pose_fcm(fcms=list_fcm, title=title, body=text)
                                background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                                
                            else:
                                text = "{:.2f}%".format(prob)
                                # text = "{:.2f}%".format(100-prob)
                                background_color = (0, 255, 0)      # Màu xanh (B, G, R)                   
                        else:
                            text = 'Trộm bẻ khoá' + " {:.2f}%".format(prob)
                            # text = "{:.2f}%".format(prob)
                            save_notification(frame, camera_id, title=title, body=text) # Lưu lại thông tin cảnh báo
                            # pose_fcm(fcms=list_fcm, title=title, body=text)
                            background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                            for a in range(10):
                                print('Trộm bẻ khoá')
                    elif prob >= thres and label=='climbing':
                        if r_queue:
                            
                            if result_queue(q_climbing, True):
                                text = 'Trộm leo rào' + " {:.2f}%".format(prob)
                                # text = "{:.2f}%".format(prob)
                                save_notification(frame, camera_id, title=title, body=text) # Lưu lại thông tin cảnh báo
                                pose_fcm(fcms=list_fcm, title=title, body=text)
                                background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                                
                            else:
                                text = "{:.2f}%".format(prob)
                                # text = "{:.2f}%".format(100-prob)
                                background_color = (0, 255, 0)      # Màu xanh (B, G, R)             
                        else:
                            text = 'Trộm leo rào' + " {:.2f}%".format(prob)
                            # text = "{:.2f}%".format(prob)
                            save_notification(frame, camera_id, title=title, body=text) # Lưu lại thông tin cảnh báo
                            # pose_fcm(fcms=list_fcm, title=title, body=text)
                            background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                            for a in range(10):
                                print('Trộm leo rào')
                    else:
                        text = 'None' + " {:.2f}%".format(prob)
                        # text = 'None' + " {:.2f}%".format(100-prob)
                        background_color = (0, 255, 0)      # Màu xanh (B, G, R)
                    
    #===========================================================================================================#                    
                    # Thiết lập font chữ
                    font_path = 'arial.ttf'
                    font_size = 40
                    font_color = (255, 255, 255)  # Màu trắng (B, G, R)
                    font = ImageFont.truetype(font_path, font_size)

                    
                    # Vẽ khung chữ nhật
                    left = int(results[0].boxes.data[0][0])
                    top = int(results[0].boxes.data[0][1])
                    right = int(results[0].boxes.data[0][2])
                    bottom = int(results[0].boxes.data[0][3])
                    cv2.rectangle(annotated_frame, (left,top), (right,bottom), background_color, thickness=3, lineType=cv2.LINE_AA)
                    
                    # Tạo một ảnh PIL từ hình ảnh Numpy
                    pil_image = Image.fromarray(annotated_frame)

                    # Tạo đối tượng vẽ trên ảnh PIL
                    draw = ImageDraw.Draw(pil_image)
                    
                    # Vẽ nền cho text 
                    text_width, text_height = draw.textsize(text, font=font)
                    left = int(results[0].boxes.data[0][0])-2
                    top = int(results[0].boxes.data[0][1])-text_height-3
                    rectangle_position = (left, top, left + text_width, top + text_height)
                    draw.rectangle(rectangle_position, fill=background_color)
                    
                    text_position = (left, top)
                    # Vẽ văn bản màu đỏ
                    draw.text(text_position, text, font=font, fill=font_color)
                    # Chuyển đổi ảnh PIL thành ảnh Numpy
                    annotated_frame = np.array(pil_image)
                    
            # except:
            #     pass
    #===========================================================================================================#            
            frame = cv2.resize(annotated_frame, 
                                (int((annotated_frame.shape[1])*scale),int((annotated_frame.shape[0])*scale)))
            cv2.imshow('YOLOv8 Inference', frame)
                
            # except:
            #     continue
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except:
        fresh.release()     
    fresh.release()
    cv2.destroyAllWindows()

while True:
    try:
        pose_cls_video(url = 'rtsp://admin:NuQuynhAnh@cam14423linhdong.smartddns.tv:1554/cam/realmonitor?channel=1&subtype=0&unicast=true')
    except:
        print("Lỗi nằm ngoài vòng WHILE !!!")
        pass

# pose_cls_video(url = 'rtsp://admin:NuQuynhAnh@cam14423linhdong.smartddns.tv:1554/cam/realmonitor?channel=1&subtype=0&unicast=true')
# pose_cls_video(url = 'rtsp://admin:Dat1qazxsw2@192.168.6.100:1554/h264_stream')
# pose_cls_video(url = 'datatest/12 - Trim.mp4')