import cv2, requests, base64, json, datetime, time
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from queue import Queue
from keras.models import load_model
from overstepframe import FreshestFrame
from inside_the_box import inside_the_box
from firebase_admin import credentials, messaging
#------------------------------------ LOAD MODEL ------------------------------------
from ultralytics import YOLO
model = YOLO('YOLOv8n-pose.pt')
pose_cls = load_model('pose_cls_v3.h5')

#------------------------------------ PARAMETERS ------------------------------------

humanpose_conf = 80
queue_len = 60
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

def save_alert_img(frame, camera_id, title, body):
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

def get_camera_data():
    
    # Lấy ra CameraID, HomeID, CameraName, RTSP, LockpickingArea, ClimbingArea
    api_url = 'http://'+ip+':5001/api/pose/get-camera-data'
    data = {
        'key': '5c1f45bde9d2aff92e03acbac0b6d49f6410ca490c1fe85a082650ee9c23f63d',
    }
    response = requests.post(api_url, json=data)
    cam_list = json.loads(response.text)
    
    """
    *  Lặp qua từng Dict trong "cam_list" và lấy danh sách FCM cần gửi thông báo, thêm vào trường FCM của dict đó
    *  Phải kiểm tra có cùng HomeID với cam trước đó k, nếu trùng k cần gọi lại "api/notification/get-fcm-to-send"
    """
    
    homeid_fcm_list = [] # List gồm các HomeID và các FCM tương ứng với HomeID đó
    
    # Lặp qua từng dict trong cam_list
    for cam in cam_list:
        call_api = True
        # Lặp qua từng dict trong homeid_fcm
        for i in homeid_fcm_list:
            # Nếu HomeID của cam nằm trong dict đã lấy FCM thì dùng lại, k cần gọi API
            if cam['HomeID']==i['HomeID']:
                cam['FCM'] = i['FCM']
                call_api = False
                break
            
        if call_api:
            cam['FCM'] = get_fcm_to_send(cam['CameraID'])
            # Thêm HomeID và các FCM tương ứng để biết đã lấy FCM của căn này rồi
            homeid_fcm_list.append({
                                    'HomeID': cam['HomeID'],
                                    'FCM': cam['FCM'],
                                    })
            
    return cam_list

def drawbox(frame, points):
    # Draw Box
    for p1, p2 in [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]:
        pt1 = tuple(points[p1])
        pt2 = tuple(points[p2])
        cv2.line(frame, pt1, pt2, (0, 0, 255), 5)
    return frame
 
def pose_cls_video():
    
    cam_data = get_camera_data()
    
    # url = [
    #         'rtsp://admin:NuQuynhAnh@cam14423linhdong.smartddns.tv:1554/cam/realmonitor?channel=1&subtype=0&unicast=true',        # Cam 1
    #         # 'rtsp://admin:Admin123@mtkhp2408.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true',                # Cam 2
    #         # 'rtsp://admin:Admin123@mtkhp2420.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true',                # Cam 3
    #         # 'rtsp://admin:Vinaai!123@py1ai.cameraddns.net:5543/cam/realmonitor?channel=1&subtype=0&unicast=true',                 # Cam 4
    #         # 'rtsp://admin:Vinaai!123@py1ai.cameraddns.net:5541/cam/realmonitor?channel=1&subtype=0&unicast=true',                 # Cam 5
    #         # 'rtsp://admin:Vinaai!123@py2ai.cameraddns.net:5541/cam/realmonitor?channel=1&subtype=0&unicast=true',                 # Cam 6
    #         # 'rtsp://admin:Vinaai!123@py2ai.cameraddns.net:5543/cam/realmonitor?channel=1&subtype=0&unicast=true',                 # Cam 7
    #         'rtsp://admin:Vinaai!123@py2ai.cameraddns.net:5545/cam/realmonitor?channel=1&subtype=0&unicast=true',                 # Cam 8
    #         ]
    
    url, lockpicking_area, climbing_area, fcm_list, camera_name, t_oldframe = [], [], [], [], [], []
    # Bỏ qua các cam chưa nhập LockpickingArea & ClimbingArea
    for cam in cam_data:
        if (cam['LockpickingArea'] is not None) or (cam['ClimbingArea'] is not None):
            url.append(cam['RTSP'])
            lockpicking_area.append(cam['LockpickingArea'])
            climbing_area.append(cam['ClimbingArea'])
            fcm_list.append(cam['FCM'])
            camera_name.append(cam['CameraName'])
            t_oldframe.append(None)
    
    #------------------------------------ FRESHEST FRAME ------------------------------------
    fresh, frame, cnt, first_frame, second_frame, None_frame, cam_name, q_lockpicking, q_climbing= [], [], [], [], [], [], [], [], []
    for i in range(len(url)):
        fresh.append(FreshestFrame(cv2.VideoCapture(url[i])))
        frame.append(object())
        cnt.append(0)
        first_frame.append(None)
        second_frame.append(None)
        None_frame.append(0)
        cam_name.append(f'Cam {i+4}')
        
        q_lockpicking.append(Queue(maxsize=queue_len))
        q_climbing.append(Queue(maxsize=queue_len))
        
    scale = 0.5
    data = {
        'key1': 'value1',
        'key2': 'value2'
    }
    thres = 80
    input = None
    
    label_mapping = {
        0: 'none',
        1: 'lockpicking',
        2: 'climbing'
        }

#===========================================================================================================#
    # try:
    while True:
        # Release FreshestFrame objects every 10 minutes
        t = datetime.datetime.now()
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
            
            try:
                frame[CC] = cv2.resize(frame[CC], (1920,1080))
                frame_width = frame[CC].shape[1]
            except:
                continue
            
            prob = 0
            
            fps = 1/(timer-t_oldframe[CC])
            t_oldframe[CC] = timer  
                        
            results = model(frame[CC], save=False)
            annotated_frame = results[0].plot(boxes=False)
            
            keypoints_arrs = results[0].keypoints.data.numpy()
            #===========================================================================================================#      
            # Nếu không phát hiện người
            if len(results[0].boxes.data)==0: 
                result_queue(q_lockpicking[CC], False)
                result_queue(q_climbing[CC], False)
            # Nếu phát hiện người
            else:
                # Lặp qua từng khung xương:
                for skeleton in range(keypoints_arrs):
                    # Nếu là con người (so sánh với humanpose_conf)
                    if float(results[0].boxes.data[skeleton][4]*100)>humanpose_conf:
                        # Kiểm tra vị trí cả 2 tay có nằm trong vùng cần kiểm tra không
                        left_hand = keypoints_arrs[skeleton,9,:2]
                        right_hand = keypoints_arrs[skeleton,10,:2]
                        
                        
                        #===========================================================================================================#
                        if lockpicking_area[CC] is not None:
                            for point in lockpicking_area[CC]:
                                # Kiểm tra nếu k có thông tin vùng Mở khóa, break
                                if lockpicking_area[CC][point] is None:
                                    break
                                
                                # Nếu có thì kiểm tra 2 vị trí 2 tay với các vùng Mở khóa
                                if (inside_the_box(left_hand,lockpicking_area[CC][point]) and inside_the_box(right_hand,lockpicking_area[CC][point])):
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
                                    #===============================#
                                    # Nếu phát hiện mở khóa
                                    title= camera_name[CC] + " - Mở khóa"
                                    if prob >= thres and label=='lockpicking':
                                        if result_queue(q_lockpicking[CC], True):
                                            text = 'Mở khoá' + " ({:.2f}%)".format(prob)
                                            post_alert(fcms=fcm_list[CC], title=title, body=text)
                                            save_alert_img(annotated_frame, camera_id=get_camera_id(url[CC]), title=title, body=text) # Lưu lại thông tin cảnh báo
                                            background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                                        else:
                                            text = "({:.2f}%)".format(prob)
                                            background_color = (0, 255, 0)      # Màu xanh (B, G, R)
                                    else:
                                        result_queue(q_lockpicking[CC], False)
                                        text = 'Bình thường' + " ({:.2f}%)".format(prob)
                                        background_color = (0, 255, 0)      # Màu xanh (B, G, R)
                                    
                                    #===============================#                    
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
                                
                                else:
                                    result_queue(q_lockpicking, False)
                                    
                        
                        #===========================================================================================================# 
                        if climbing_area[CC] is not None:
                            for point in climbing_area[CC]:
                                # Kiểm tra nếu k có thông tin vùng Mở khóa, break
                                if climbing_area[CC][point] is None:
                                    break
                                # Vẽ vùng cảnh báo
                                annotated_frame = drawbox(annotated_frame, climbing_area[CC][point])
                                # Nếu có thì kiểm tra 2 vị trí 2 tay với các vùng Leo rào
                                if (inside_the_box(left_hand,climbing_area[CC][point]) and inside_the_box(right_hand,climbing_area[CC][point])):
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
                                    #===============================#
                                    # Nếu phát hiện mở khóa
                                    title = camera_name[CC] + " - Leo rào"
                                    if prob >= thres and label=='climbing':
                                        if result_queue(q_climbing[CC], True):
                                            text = 'Leo rào' + " ({:.2f}%)".format(prob)
                                            post_alert(fcms=fcm_list[CC], title=title, body=text)
                                            save_alert_img(annotated_frame, camera_id=get_camera_id(url[CC]), title=title, body=text)
                                            background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                                        else:
                                            text = "({:.2f}%)".format(prob)
                                            background_color = (0, 255, 0)      # Màu xanh (B, G, R)
                                    else:
                                        result_queue(q_climbing[CC], False)
                                        text = 'Bình thường' + " ({:.2f}%)".format(prob)
                                        background_color = (0, 255, 0)      # Màu xanh (B, G, R)
                                    
                                    #===============================#                    
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
                                
                                else:
                                    result_queue(q_climbing, False)
            #===========================================================================================================#
            frame[CC] = annotated_frame
            
            # Hiện FPS
            cv2.putText(frame[CC], "fps: {:.2f}".format(fps), (20,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 155, 255), 2)
            
            # Vẽ vùng cảnh báo
            if lockpicking_area[CC] is not None:
                for point in lockpicking_area[CC]:
                    frame[CC] = drawbox(frame[CC], lockpicking_area[CC][point])
            if climbing_area[CC] is not None:
                for point in climbing_area[CC]:
                    frame[CC] = drawbox(frame[CC], climbing_area[CC][point])
                
            cv2.imshow(camera_name[CC], 
                        cv2.resize(frame[CC], (int((frame[CC].shape[1])*scale),int((frame[CC].shape[0])*scale))))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                
    # except Exception as e:
    #     print(e)
    #     for CC in range(len(url)):
    #         fresh[CC].release()
            
    for CC in range(len(url)):
        fresh[CC].release()
        
    cv2.destroyAllWindows()
    # print('Restarting...')
    
    
pose_cls_video()
# while True:
#     try:
#         pose_cls_video()
#     except:
#         print("Lỗi nằm ngoài vòng WHILE !!!")
#         pass
