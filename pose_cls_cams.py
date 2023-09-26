import cv2, requests, base64, json, datetime, time
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from queue import Queue
from tensorflow.keras.models import load_model
from overstepframe import FreshestFrame
from inside_the_box import inside_the_box
from datetime import datetime
from firebase_admin import credentials, messaging
# from firebase_admin import credentials, messaging
#------------------------------------ LOAD MODEL ------------------------------------
from ultralytics import YOLO
model = YOLO('YOLOv8n-pose.pt')
pose_cls = load_model('pose_cls_v3.h5')

#------------------------------------ PARAMETERS ------------------------------------
points = {
    'Cam1':
    {
        'A': [800, 0],
        'B': [1005, 0],
        'C': [875, 1080],
        'D': [500, 1080],
    },
    'Cam8':
    {
        'A': [1230, 600],
        'B': [1320, 710],
        'C': [1230, 880],
        'D': [1145, 770],
    },
}
cam_name_list = ['Cam1', 'Cam8']
# cam_name_list = ['Cam8']
humanpose_conf = 80
queue_len = 90
# ip = '192.168.6.17'
ip = '125.253.117.120'
# fcm = 'dSZbYbanSl-pIr8eBcL2KN:APA91bHsX7uv4J2TdaoEbxsZg9y3U_Y54QWkkBCw8Xko8It0-w5XbFY5ae6VIiM1iT_r-xDyzF_gq0jCorYx5aBN7OL49ULuC9ay5n1dUmCKO0X3HYa5Dv3X8aV7faym47ZcJWPBYhwo'
# fcm = 'dwiwc-hqSSqauJ8sXPQkCM:APA91bH0kZuNqUxm1Q-vOuHgHvpfiEAA6gmkbAenmG_6pbFwZ-0QwrGg03sTaVBfKvySGSRk2w24mM4zGRNPjxpNHT9wXgywsDZGjumWPyYoxr-LzQ6PoIqb0Bl9HTOFIC522SDkK8f6'
# fcm = 'fNM-TsWQSqmLsBUg9HejwR:APA91bG8owUPOWHA0mCHs4f8Pi3Pqtus0iLszlPajoeX2nQtYkQ8v6LpDe3n8b1zDI2FLxUKOs_fosMrkc-7TA_bN2kY9B8GGd1xe89GQESaL6Ir5Qlz3-zA2uFEe4Xd-KB55PEtxS32'

#------------------------------------------------------------------------------------------------------------
def remove_duplicates_and_none(input_list):
    result = []
    
    for sublist in input_list:
        unique_sublist = []
        for element in sublist:
            if (element is None) or (element in unique_sublist):
                continue
            else:
                unique_sublist.append(element)
        
        result.append(unique_sublist)
    
    return result

def result_queue(q,value):
    if q.full():
        q.get()  # Lấy giá trị đầu tiên khi Queue đã đủ 60 phần tử
    q.put(value)  # Thêm giá trị mới vào Queue
    if q.queue.count(True) >= (queue_len-5):  # Kiểm tra nếu có ít nhất 50 True trong Queue
        if value:
            return True
        else:
            return False
    else:
        return False

def get_camera_id(rtsp):
    api_url = 'http://'+ip+':5001/api/notification/get-camera-id'
    data = {
        'key': '5c1f45bde9d2aff92e03acbac0b6d49f6410ca490c1fe85a082650ee9c23f63d',
        'rtsp': rtsp,

    }
    response = requests.post(api_url, json=data)
    return response.json()['message']

def save_alert_img(frame, camera_id, title, body, formatted_time):
    api_url = 'http://'+ip+':5001/api/notification/save'
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
        'formatted_time': formatted_time,
    }
    response = requests.post(api_url, json=data)
    print(response.json())
    
def get_fcm_to_send(camera_id):
    api_url = 'http://'+ip+':5001/api/notification/get-fcm-to-send'
    data = {
        'key': '5c1f45bde9d2aff92e03acbac0b6d49f6410ca490c1fe85a082650ee9c23f63d',
        'camera_id': camera_id,
    }
    response = requests.post(api_url, json=data)
    return json.loads(response.text)

def post_alert(fcms, title, body, data=None):
    message = messaging.MulticastMessage( 
                            notification = messaging.Notification( title=title, body=body), 
                            # android=messaging.AndroidConfig( priority='high', notification=messaging.AndroidNotification( sound=sound_path, image=image_url ), ), 
                            # apns=messaging.APNSConfig( payload=messaging.APNSPayload( aps=messaging.Aps( sound=sound_path ), ), ), 
                            tokens=fcms
                            )
    # Gửi thông báo đến thiết bị cụ thể
    response = messaging.send_multicast(message)
    print(f"Failure Count: {response.failure_count}")
    return response.failure_count
    
def pose_cls_video(r_queue=False):
    # fcm = 'dSZbYbanSl-pIr8eBcL2KN:APA91bHsX7uv4J2TdaoEbxsZg9y3U_Y54QWkkBCw8Xko8It0-w5XbFY5ae6VIiM1iT_r-xDyzF_gq0jCorYx5aBN7OL49ULuC9ay5n1dUmCKO0X3HYa5Dv3X8aV7faym47ZcJWPBYhwo'
    # fcm = 'dwiwc-hqSSqauJ8sXPQkCM:APA91bH0kZuNqUxm1Q-vOuHgHvpfiEAA6gmkbAenmG_6pbFwZ-0QwrGg03sTaVBfKvySGSRk2w24mM4zGRNPjxpNHT9wXgywsDZGjumWPyYoxr-LzQ6PoIqb0Bl9HTOFIC522SDkK8f6'
    # fcm = 'fNM-TsWQSqmLsBUg9HejwR:APA91bG8owUPOWHA0mCHs4f8Pi3Pqtus0iLszlPajoeX2nQtYkQ8v6LpDe3n8b1zDI2FLxUKOs_fosMrkc-7TA_bN2kY9B8GGd1xe89GQESaL6Ir5Qlz3-zA2uFEe4Xd-KB55PEtxS32'
    # url = 'rtsp://admin:NuQuynhAnh@cam14423linhdong.smartddns.tv:1554/cam/realmonitor?channel=1&subtype=0&unicast=true' # Cam 1
    # url = 'rtsp://admin:Admin123@mtkhp2408.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true'         # Cam 2
    # url = 'rtsp://admin:Admin123@mtkhp2420.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true'         # Cam 3

    url = [
            'rtsp://admin:NuQuynhAnh@cam14423linhdong.smartddns.tv:1554/cam/realmonitor?channel=1&subtype=0&unicast=true',        # Cam 1
            # 'rtsp://admin:Admin123@mtkhp2408.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true',                # Cam 2
            # 'rtsp://admin:Admin123@mtkhp2420.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true',                # Cam 3
            # 'rtsp://admin:Vinaai!123@py1ai.cameraddns.net:5543/cam/realmonitor?channel=1&subtype=0&unicast=true',                 # Cam 4
            # 'rtsp://admin:Vinaai!123@py1ai.cameraddns.net:5541/cam/realmonitor?channel=1&subtype=0&unicast=true',                 # Cam 5
            # 'rtsp://admin:Vinaai!123@py2ai.cameraddns.net:5541/cam/realmonitor?channel=1&subtype=0&unicast=true',                 # Cam 6
            # 'rtsp://admin:Vinaai!123@py2ai.cameraddns.net:5543/cam/realmonitor?channel=1&subtype=0&unicast=true',                 # Cam 7
            'rtsp://admin:Vinaai!123@py2ai.cameraddns.net:5545/cam/realmonitor?channel=1&subtype=0&unicast=true',                 # Cam 8
            ]
    #------------------------------------ FRESHEST FRAME ------------------------------------
    fresh, frame, cnt, first_frame, second_frame, None_frame, cam_name, list_fcm, t_oldframe = [], [], [], [], [], [], [], [], []
    for i in range(len(url)):
        fresh.append(object())
        frame.append(object())
        cnt.append(0)
        first_frame.append(None)
        second_frame.append(None)
        t_oldframe.append(None)
        None_frame.append(0)
        cam_name.append(f'Cam {i+4}')
        
        camera_id = get_camera_id(url[i])
        list_fcm.append(get_fcm_to_send(camera_id))
        
        fresh[i]=FreshestFrame(cv2.VideoCapture(url[i]))
        
    list_fcm = remove_duplicates_and_none(list_fcm)
    scale = 0.5
    data = {
        'key1': 'value1',
        'key2': 'value2'
    }
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
            # Release FreshestFrame objects every 10 minutes
            t = datetime.now()
            if (t.minute % 10 == 0) and t.second<2:
                raise Exception("Restarting...")
            
            for CC in range(len(url)):
                cnt[CC],frame[CC] = fresh[CC].read(seqnumber=cnt[CC]+1, timeout=5)
                if not cnt[CC]:
                    print(f"Timeout, can't read new frame of cam {CC}!")
                    raise Exception()
                
                if None_frame[CC]>5:
                    print("Cannot read frame from camera!")
                    raise Exception()
                
                timer =time.time()
                if t_oldframe[CC] is None:
                    t_oldframe[CC] = timer
                    
                if first_frame[CC] is None:
                    first_frame[CC] = frame[CC]
                    None_frame[CC]+=1
                    continue
                
                current_time = datetime.now()
                formatted_time = current_time.strftime("%d-%m-%Y %Hh%M'%S\"")
                
                try:
                    frame[CC] = cv2.resize(frame[CC], (1920,1080))
                    frame_width = frame[CC].shape[1]
                except:
                    continue
                
                prob = 0
                fps = 1/(timer-t_oldframe[CC])

                # results = model(cv2.resize(frame, (640,640)), save=False)
                results = model(frame[CC], save=False)
                annotated_frame = results[0].plot(boxes=False)
                
                keypoints_arrs = results[0].keypoints.data.numpy()
                #===========================================================================================================#            

                if len(results[0].boxes.data)>0:
                    for skeleton in range(len(keypoints_arrs)):
                        # if float(results[0].boxes.data[0][4]*100)>humanpose_conf:
                        left_hand = keypoints_arrs[skeleton,9,:2]
                        right_hand = keypoints_arrs[skeleton,10,:2]
                        if (float(results[0].boxes.data[skeleton][4]*100)>humanpose_conf 
                            and inside_the_box(left_hand,points[cam_name_list[CC]]) and inside_the_box(right_hand,points[cam_name_list[CC]])):
                            
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
                            # current_time = datetime.now()
                            title_save_ntf=f"Có ăn trộm"
                            # title_fcm=f"{cam_name_list[CC]} - " + current_time.strftime("%Hh%M'%S\" %d-%m-%Y") 
                            if prob >= thres and label=='lockpicking':
                                title_fcm=f"{cam_name_list[CC]} - Mở khóa"
                                if r_queue:
                                    if result_queue(q_lockpicking, True):
                                        text = 'Hành vi mở khoá' + " ({:.2f}%)".format(prob)
                                        # save_alert_img(frame[CC], camera_id=get_camera_id(url[CC]), title=title, body=text) # Lưu lại thông tin cảnh báo
                                        post_alert(fcms=list_fcm[CC], title=title_fcm, body=text)
                                        # text = "({:.2f}%)".format(prob)
                                        background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                                    else:
                                        text = "({:.2f}%)".format(prob)
                                        background_color = (0, 255, 0)      # Màu xanh (B, G, R)
                                else:
                                    text = 'Hành vi mở khoá' + " ({:.2f}%)".format(prob)
                                    # save_alert_img(frame[CC], camera_id=get_camera_id(url[CC]), title=title, body=text) # Lưu lại thông tin cảnh báo
                                    post_alert(fcms=list_fcm[CC], title=title_fcm, body=text)
                                    # text = "({:.2f}%)".format(prob)
                                    background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                            elif prob >= thres and label=='climbing':
                                title_fcm=f"{cam_name_list[CC]} - Leo rào"
                                if r_queue:
                                    if result_queue(q_climbing, True):
                                        text = 'Hành vi leo rào' + " ({:.2f}%)".format(prob)
                                        # save_alert_img(frame[CC], camera_id=get_camera_id(url[CC]), title=title, body=text) # Lưu lại thông tin cảnh báo
                                        post_alert(fcms=list_fcm[CC], title=title_fcm, body=text)
                                        # text = "({:.2f}%)".format(prob)
                                        background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                                    else:
                                        text = "({:.2f}%)".format(prob)
                                        background_color = (0, 255, 0)      # Màu xanh (B, G, R)
                                else:
                                    text = 'Hành vi leo rào' + " ({:.2f}%)".format(prob)
                                    # save_alert_img(frame[CC], camera_id=get_camera_id(url[CC]), title=title, body=text) # Lưu lại thông tin cảnh báo
                                    post_alert(fcms=list_fcm[CC], title=title_fcm, body=text)
                                    # text = "({:.2f}%)".format(prob)
                                    background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                            else:
                                if r_queue:
                                    result_queue(q_lockpicking, False)
                                    result_queue(q_climbing, False)
                                text = 'Bình thường' + " ({:.2f}%)".format(prob)
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
                            
                            if prob >= thres and label=='lockpicking':
                                if r_queue:
                                    if result_queue(q_lockpicking, True):
                                        save_alert_img(annotated_frame, camera_id=get_camera_id(url[CC]), title=title_save_ntf, body=text, formatted_time=formatted_time) # Lưu lại thông tin cảnh báo
                                else:
                                    save_alert_img(annotated_frame, camera_id=get_camera_id(url[CC]), title=title_save_ntf, body=text, formatted_time=formatted_time) # Lưu lại thông tin cảnh báo
                            elif prob >= thres and label=='climbing':
                                if r_queue:
                                    if result_queue(q_climbing, True):
                                        save_alert_img(annotated_frame, camera_id=get_camera_id(url[CC]), title=title_save_ntf, body=text, formatted_time=formatted_time) # Lưu lại thông tin cảnh báo
                                else:
                                    save_alert_img(annotated_frame, camera_id=get_camera_id(url[CC]), title=title_save_ntf, body=text, formatted_time=formatted_time) # Lưu lại thông tin cảnh báo
                        
                        else:
                            result_queue(q_lockpicking, False)
                            result_queue(q_climbing, False)
                else:
                    if result_queue:
                        result_queue(q_lockpicking, False)
                        result_queue(q_climbing, False)
                #===========================================================================================================#
                frame[CC] = annotated_frame
                # Draw Box
                for p1, p2 in [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]:
                    pt1 = tuple(points[cam_name_list[CC]][p1])
                    pt2 = tuple(points[cam_name_list[CC]][p2])
                    cv2.line(frame[CC], pt1, pt2, (0, 0, 255), 5)
                
                # Hiện FPS
                cv2.putText(frame[CC], "fps: {:.2f}".format(fps), (20,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 155, 255), 2)
                        
                frame[CC] = cv2.resize(frame[CC], 
                                    (int((frame[CC].shape[1])*scale),int((frame[CC].shape[0])*scale)))
                
                # Đặt văn bản lên khung hình

                # cv2.putText(frame[CC], f"{human_skeleton} human", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.imshow('YOLOv8 Inference', frame[CC])
                cv2.imshow(cam_name_list[CC], frame[CC])
                    
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(e)
        for CC in range(len(url)):
            fresh[CC].release()
            
    for CC in range(len(url)):
        fresh[CC].release()
        
    cv2.destroyAllWindows()
    # print('Restarting...')

while True:
    try:
        pose_cls_video()
    except:
        print("Lỗi nằm ngoài vòng WHILE !!!")
        pass


# for i in range(3):
#     post_alert(fcms=['epj7mtcHRiusC9Nm0xU--6:APA91bGXY3WN6ikQWVl9rGO81uZUdluFiPVK6kR2mA2__E1PPi2tDDa3uC7NvfW1NAYOS4Qoz9Tw5TPT4Q9yNfPILTstCowwboArZOYWV6yU5Am_5rvMDj7qXl8MEo99jRC7Sxif03WF',
#                     'dHep6NaQS_O4Nd--QnXYLg:APA91bFyGx9K3H_fWapfKxYZPY8MhVbBIPeN_9M97WkFM7qEZmucNdonMvdd57dkWjZrFV61ubAPEWD0-VyJEYH8l8nIVhbgSSoClpP6cXUNX_vLxAx019TwmMcIjr5Y8btn9L_Sb7Lu',
#                     'eEoHWmwGQuCrLz9ttBqZ2E:APA91bHyQ7nYDApqwARdHozfDZI1UvdYJZLJ3AExac11JTU3XjKTYBZdCzVncbF4F8fb_YU_H8smAdBshE-fhsJyEJIWfKNhYR-PhOvCZmbluXRzhYbPzUphS8B-pXlg4omS7HObKIvC',], 
#             title="AAAAAAAAAA", 
#             body="Adudu Adudu dududu", 
#             data=None)
#     time.sleep(1)


# pose_cls_video(url = 'rtsp://admin:NuQuynhAnh@cam14423linhdong.smartddns.tv:1554/cam/realmonitor?channel=1&subtype=0&unicast=true')
# pose_cls_video(url = 'rtsp://admin:Dat1qazxsw2@192.168.6.100:1554/h264_stream')
# pose_cls_video(url = 'datatest/12 - Trim.mp4')