import cv2, requests, base64, json, datetime, time, pyodbc, dlib, threading, firebase_admin
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from queue import Queue
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine, euclidean
from overstepframe import FreshestFrame
from inside_the_box import inside_the_box
from prepare_data import download_hinhtrain, load_hinhtrain
from remote_lock import get_accesstoken, lock, unlock
from face_detect import detect_face
from facereg_model import loadVggFaceModel
from eyeblink import predictor_eye, get_blinking_ratio
from padding_image import padding
from firebase_admin import credentials, messaging

detector = cv2.dnn.readNetFromCaffe("pre_model/deploy.prototxt","pre_model/res10_300x300_ssd_iter_140000.caffemodel")
vgg_model = loadVggFaceModel()
# from firebase_admin import credentials, messaging
#------------------------------------ LOAD MODEL ------------------------------------
from ultralytics import YOLO
pose_model = YOLO('YOLOv8n-pose.pt')
fire_model = YOLO('FireSmokePerson_v2.pt')
pose_cls = load_model('pose_cls_v3.h5')
cred = credentials.Certificate('test/ngocvinaai-firebase-adminsdk-57cev-b988d1a956.json')
firebase_admin.initialize_app(cred)
#------------------------------------ PARAMETERS ------------------------------------

humanpose_conf = 80
queue_len = 10
# ip = '192.168.6.17'
ip = '125.253.117.120'

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
    if q.queue.count(True) >= (queue_len-5):  # Kiểm tra nếu có ít nhất 5 True trong Queue
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

def save_ntf_img(frame, camera_id, title, body, notification_type, formatted_time):
    api_url = 'http://'+ip+':5001/api/notification/save'
    # Chuyển đổi dữ liệu ảnh thành chuỗi base64
    _, image_data = cv2.imencode('.jpg', frame)
    base64_image = base64.b64encode(image_data).decode("utf-8")
    
    data = {
        'key': '5c1f45bde9d2aff92e03acbac0b6d49f6410ca490c1fe85a082650ee9c23f63d',
        'camera_id': camera_id,
        'notification_type': notification_type,
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

def post_alert(fcm_list, title, body, data=None):
    message = messaging.MulticastMessage( 
                            notification = messaging.Notification( title=title, body=body), 
                            android=messaging.AndroidConfig( priority='high', notification=messaging.AndroidNotification(sound='res_sound45', priority='max')), 
                            # apns=messaging.APNSConfig( payload=messaging.APNSPayload( aps=messaging.Aps( sound=sound_path ), ), ), 
                            tokens=fcm_list
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

def base64_to_array(anh_base64):
        try:
            img_arr = np.frombuffer(base64.b64decode(anh_base64), dtype=np.uint8)
            img_arr = cv2.imdecode(img_arr, cv2.IMREAD_ANYCOLOR)
        except:
            return "Không chuyển được ảnh base64 sang array"
        return img_arr

def post_ntf_knownperson_detected(faceid, cameraname, fcm_list, formatted_time):
    connection = pyodbc.connect("Driver={SQL Server};"
                                "Server=112.78.15.3;"
                                "Database=VinaAIAPP;"
                                "uid=ngoi;"
                                "pwd=admin123;")
    cursor = connection.cursor()
    
    cursor.execute("SELECT FaceName FROM FaceRegData WHERE FaceID = ?", (faceid,))
    facename = cursor.fetchone().FaceName

    body = f"Đã phát hiện {facename} ở camera {cameraname} lúc {formatted_time}"

    post_alert(fcm_list=fcm_list, title="Phát hiện người quen", body=body)

def predict_fire(frame, annotated_frame, current_time, q_fire, q_smoke, fcm_list, url):
    label_mapping = {
    0: "fire",
    1: "person",
    2: "smoke"
    }
    max_prob_fire = None
    max_prob_smoke = None
    max_prob_person = None
    formatted_time = current_time.strftime("%d-%m-%Y %Hh%M'%S\"")
    formatted_time_ntf = ' lúc ' + current_time.strftime("%Hh%M'%S\" %d-%m-%Y")
    
    copy_frame = frame.copy()
    results = fire_model.predict(source=copy_frame, conf=0.25, save=False)
    
    for result in results[0].boxes.data:
        # Thiết lập font chữ
        font_path = 'arial.ttf'
        font_size = 40
        font_color = (255, 255, 255)  # Màu trắng (B, G, R)
        font = ImageFont.truetype(font_path, font_size)

        # Vẽ khung chữ nhật
        left = int(result[0])
        top = int(result[1])
        right = int(result[2])
        bottom = int(result[3])
        prob = int(result[4])
        label = int(result[5])
        label = label_mapping.get(label, "unknown")
        
        if label=='fire':
            background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
            text = "Lửa ({:.2f}%)".format(prob)
            if max_prob_fire==None or max_prob_fire<prob:
                max_prob_fire=prob
                
        elif label=='person':
            background_color = (0, 255, 0)      # Màu xanh (B, G, R)
            text = "Có người ({:.2f}%)".format(prob)
            if max_prob_person==None or max_prob_person<prob:
                max_prob_person=prob
                
        elif label=='smoke':
            background_color = (224, 144, 139)
            text = "Khói ({:.2f}%)".format(prob)
            if max_prob_smoke==None or max_prob_smoke<prob:
                max_prob_smoke=prob      
        
        cv2.rectangle(annotated_frame, (left,top), (right,bottom), background_color, thickness=3, lineType=cv2.LINE_AA)
            
        # Tạo một ảnh PIL từ hình ảnh Numpy
        pil_image = Image.fromarray(annotated_frame)

        # Tạo đối tượng vẽ trên ảnh PIL
        draw = ImageDraw.Draw(pil_image)
        
        # Vẽ nền cho text 
        text_width, text_height = draw.textsize(text, font=font)
        t_left = int(result[0])-2
        t_top = int(result[1])-text_height-3
        rectangle_position = (t_left, t_top, t_left + text_width, t_top + text_height)
        draw.rectangle(rectangle_position, fill=background_color)
        
        text_position = (t_left, t_top)
        # Vẽ văn bản màu đỏ
        draw.text(text_position, text, font=font, fill=font_color)
        # Chuyển đổi ảnh PIL thành ảnh Numpy
        annotated_frame = np.array(pil_image)
    
    # Nếu có lửa mà không có người
    if max_prob_fire is not None and max_prob_person is None:
        if result_queue(q_fire, True):
            
            text = 'Có cháy' + " ({:.2f}%)".format(max_prob_fire)
            title = "Cảnh báo cháy"
            post_alert(fcm_list=fcm_list, title=title, body=text + formatted_time_ntf)
            save_ntf_img(annotated_frame, 
                        camera_id=get_camera_id(url), 
                        title=title, 
                        body=text,
                        notification_type='Fire',
                        formatted_time=formatted_time) # Lưu lại thông tin cảnh báo
    else:
        result_queue(q_fire, False)
    
    # Nếu có khói mà không có người        
    if max_prob_smoke is not None and max_prob_person is None:
        if result_queue(q_smoke, True):
            
            text = 'Có khói' + " ({:.2f}%)".format(max_prob_smoke)
            title = "Cảnh báo cháy"
            post_alert(fcm_list=fcm_list, title=title, body=text + formatted_time_ntf)
            save_ntf_img(annotated_frame, 
                        camera_id=get_camera_id(url), 
                        title=title, 
                        body=text,
                        notification_type='Fire',
                        formatted_time=formatted_time) # Lưu lại thông tin cảnh báo
    else:
        result_queue(q_smoke, False)


def detect_skeleton(frame):
    return pose_model(frame, save=False)
# face_reg_data = get_FaceRegData()

def pose_cls_video():
    download_hinhtrain()
    known_persons = load_hinhtrain()
    cam_data = get_camera_data()
    cam_data = [cam_data[9]]
    #------------------------------------ Các thông tin của Camera ------------------------------------
    url, lockpicking_area, climbing_area, fcm_list, camera_id, camera_name, homeid, lockid, related_camera_id, task = [], [], [], [], [], [], [], [], [], []
    # Bỏ qua các cam chưa nhập LockpickingArea & ClimbingArea và không có LockID
    for cam in cam_data:
        if not ((cam['LockpickingArea'] is None) and (cam['ClimbingArea'] is None) and (cam['LockID'] is None)):
            url.append(cam['RTSP'])
            lockpicking_area.append(cam['LockpickingArea'])
            climbing_area.append(cam['ClimbingArea'])
            fcm_list.append(cam['FCM'])
            camera_id.append(cam['CameraID'])
            related_camera_id.append(cam['RelatedCameraID'])
            camera_name.append(cam['CameraName'])
            homeid.append(cam['HomeID'])
            lockid.append(cam['LockID'])
            # Phân biệt cam chạy FaceID hay Pose
            task.append('Pose' if cam['LockID'] is None else 'FaceID')
            
    #------------------------------------ FRESHEST FRAME ------------------------------------
    fresh, frame, cnt, first_frame, second_frame, t_oldframe, None_frame, q_knownperson, q_lockpicking, q_climbing, lastest_detected_face, q_fire, q_smoke = [], [], [], [], [], [], [], [], [], [], [], [], []
    for i in range(len(url)):
        fresh.append(FreshestFrame(cv2.VideoCapture(url[i])))
        frame.append(object())
        cnt.append(0)
        first_frame.append(None)
        second_frame.append(None)
        t_oldframe.append(None)
        None_frame.append(0)
        lastest_detected_face.append(None)
        q_lockpicking.append(Queue(maxsize=queue_len))
        q_climbing.append(Queue(maxsize=queue_len))
        q_knownperson.append(Queue(maxsize=10))
        q_fire.append(Queue(maxsize=queue_len))
        q_smoke.append(Queue(maxsize=queue_len))
    
    #======================= Params =======================#
    scale = 0.5
    pose_thres = 80
    thresh = 127
    input = None
    flag_save_ntf_img = None
    w_box4detectface = 480
    h_box4detectface = 360
    label_mapping = {
        0: 'none',
        1: 'lockpicking',
        2: 'climbing'
        }

    #===========================================================================================================#
    cap = cv2.VideoCapture('datatest/debug1.mp4')
    if not cap.isOpened():
        print("Không thể mở video.")
        exit()
    #===========================================================================================================#
    # try:
    while True:
        # Release FreshestFrame objects every 10 minutes
        t = datetime.datetime.now()
        # if (t.minute % 10 == 0) and t.second<2:
        #     raise Exception("Restarting...")
        
        for CC in range(len(url)):
            # cnt[CC],frame[CC] = fresh[CC].read(seqnumber=cnt[CC]+1, timeout=5)
            cnt[CC],frame[CC] = cap.read()
            if not cnt[CC]:
                break
            # frame[CC] = cv2.resize(frame[CC], (1920,1080))
            if not cnt[CC]:
                print(f"Timeout, can't read new frame of cam {CC}!")
                raise Exception()
            
            if None_frame[CC]>5:
                print("Cannot read frame from camera!")
                raise Exception()
            
            # dùng để tính toán FPS
            timer =time.time()
            if t_oldframe[CC] is None:
                t_oldframe[CC] = timer
            
            # gọi lỗi nếu k đọc được frame từ camera
            if first_frame[CC] is None:
                first_frame[CC] = frame[CC].copy()
                None_frame[CC]+=1
                continue
            
            frame_width = frame[CC].shape[1]
            frame_height = frame[CC].shape[0]
            base_img = frame[CC].copy()
            prob = 0
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%d-%m-%Y %Hh%M'%S\"")
            formatted_time_ntf = ' lúc ' + current_time.strftime("%Hh%M'%S\" %d-%m-%Y")
            
            fps = 1/(timer-t_oldframe[CC])
            t_oldframe[CC] = timer
            
            first_frame[CC] = cv2.cvtColor(first_frame[CC], cv2.COLOR_BGR2GRAY)
            first_frame[CC] = cv2.threshold(first_frame[CC], thresh, 255, cv2.THRESH_BINARY)[1]

            second_frame[CC] = frame[CC].copy()
            annotated_frame = frame[CC].copy()
            second_frame[CC] = cv2.cvtColor(second_frame[CC], cv2.COLOR_BGR2GRAY)
            second_frame[CC] = cv2.threshold(second_frame[CC], thresh, 255, cv2.THRESH_BINARY)[1]

            res = cv2.absdiff(first_frame[CC], second_frame[CC])
            first_frame[CC] = frame[CC].copy()
            res = res.astype(np.uint8)
            percentage = (np.count_nonzero(res) * 100)/ res.size
            print('FRAM DIFF: ', percentage)
            
            # if percentage>=0.12:
            #=========================== Phát hiện khung xương trước ===================================#
            
            # # Tạo hai luồng riêng biệt để dự đoán với hai model khác nhau
            # fire_thread = threading.Thread(target=predict_fire, args=(frame[CC], annotated_frame, current_time, q_fire[CC], q_smoke[CC], fcm_list[CC], url[CC]))
            # pose_thread = threading.Thread(target=detect_skeleton, args=(frame[CC]))
            # fire_thread.start()
            # pose_thread.start()
            # fire_thread.join()
            # pose_thread.join()
            
            
            results = pose_model(frame[CC], save=False)
            annotated_frame = results[0].plot(boxes=False)
            keypoints_arrs = results[0].keypoints.data.numpy()

            #=========================== Chỗ này phát hiện lửa, khói ===================================#
            predict_fire(frame[CC], annotated_frame, current_time, q_fire[CC], q_smoke[CC], fcm_list[CC], url[CC])
            #===========================================================================================#
            
            # Nếu không phát hiện người
            if len(results[0].boxes.data)==0: 
                result_queue(q_lockpicking[CC], False)
                result_queue(q_climbing[CC], False)
            # Nếu phát hiện người
            else:
                flag_nohuman = True
                flag_inside_box_lockpicking = False
                flag_inside_box_climbing = False
                for skeleton in range(len(keypoints_arrs)):
                    if float(results[0].boxes.data[skeleton][4]*100)>humanpose_conf:
                        flag_nohuman = False
                        print(float(results[0].boxes.data[skeleton][4]*100))
                        #------------------------------ Tìm vị trí khuôn mặt ------------------------------
                        current_skeleton = keypoints_arrs[skeleton]
                        highestVisible = np.argmax(current_skeleton[:5,2])
                        # x1, y1 là tọa độ 1 trong 5 điểm trên đầu có Visible cao nhất
                        x1 = int(current_skeleton[highestVisible,0] - w_box4detectface/2)
                        y1 = int(current_skeleton[highestVisible,1] - h_box4detectface/2)
                        
                        # Trường hợp topleft nằm ngoài frame
                        if x1<0: x1=0
                        if y1<0: y1=0
                        if x1>frame_width-w_box4detectface: x1=frame_width-w_box4detectface
                        if y1>frame_height-h_box4detectface: y1=frame_height-h_box4detectface
                        
                        
                        # Detect Face
                        box4detectface =  base_img[y1:y1 + h_box4detectface, x1:x1 + w_box4detectface].copy()
                        original_size = box4detectface.shape # base_img.shape
                        target_size = (300, 300) # Target size của model detectface
                        # img = cv2.resize(frame, target_size)
                        aspect_ratio_x = (original_size[1] / target_size[1])
                        aspect_ratio_y = (original_size[0] / target_size[0])
                        
                        # Các tọa độ của face: top, left, right, bottom trong box 480x360
                        detections_df = detect_face(ori_img=box4detectface, 
                                                    detector=detector,)
                                                    # drawbox_on=annotated_frame[y1:y1 + h_box4detectface, x1:x1 + w_box4detectface])
                        
                        # Nếu không phát hiện có khuôn mặt nào
                        if len(detections_df)==0:
                            # Lưu lại thời gian phát hiện khuôn mặt ng quen
                            # Trong 20s sẽ k chạy nhận diện hvi
                            
                            result_queue(q_knownperson[CC], False)
                            
                            if lastest_detected_face[CC] is not None:
                                time_difference = current_time - lastest_detected_face[CC]
                                if time_difference.total_seconds() >= 20:
                                    lastest_detected_face[CC] = None
                            
                            print("NO FACE DETECTED")
                                    
                        #====================================== Nhận diện khuôn mặt ================================================#
                        # Nếu phát hiện có khuôn mặt            
                        else:
                            
                            for index_face, instance in detections_df.iterrows():
                                # confidence_score = str(round(100*instance["confidence"], 2))+" %"
                                
                                # CHUYỂN VỀ TỌA ĐỘ TRÊN ẢNH GỐC
                                
                                left = int(instance["left"]*aspect_ratio_x + x1); right = int(instance["right"]*aspect_ratio_x + x1)
                                bottom = int(instance["bottom"]*aspect_ratio_y + y1); top = int(instance["top"]*aspect_ratio_y + y1)
                                
                                # left = int(instance["left"]*aspect_ratio_x); right = int(instance["right"]*aspect_ratio_x)
                                # bottom = int(instance["bottom"]*aspect_ratio_y); top = int(instance["top"]*aspect_ratio_y)
                                
                                p2c = current_skeleton[highestVisible,:2].astype(int)
                                abcd = {
                                    'A': [left, top],
                                    'B': [right, top],
                                    'C': [right, bottom],
                                    'D': [left, bottom],
                                }
                                if not inside_the_box(point_test=p2c, points=abcd, img_h=frame_height):
                                    continue
                                
                                # Cắt ảnh khuôn mặt người để so khớp
                                detected_face = base_img[top:bottom,left:right]
                                                        
                                # Cắt ảnh khuôn mặt người để lưu lại
                                # detected_face1 = base_img[int(top*aspect_ratio_y)-20:int(bottom*aspect_ratio_y)+20,
                                #                         int(left*aspect_ratio_x)-20:int(right*aspect_ratio_x)+20]
                                # top1 = int(top*aspect_ratio_y)-20 if (int(top*aspect_ratio_y)-20)>=0 else 0
                                # bottom1 = int(bottom*aspect_ratio_y)+20 if (int(top*aspect_ratio_y)+20)<original_size[1] else original_size[1]
                                # left1 = int(left*aspect_ratio_x)-20 if (int(left*aspect_ratio_x)-20)>=0 else int(left*aspect_ratio_x)-20
                                # right1 = int(right*aspect_ratio_x)+20 if (int(right*aspect_ratio_x)+20)<original_size[0] else original_size[0]
                                # detected_face1 = base_img[top1:bottom1, left1:right1]
                            
                                ## lanmark face eyes
                                gey = dlib.rectangle(left, top, right, bottom)
                                                    
                                landmarks = predictor_eye(frame[CC], gey)
                                # Get eyes positions
                                left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks,frame[CC])
                                right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks,frame[CC])
                                blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
                                
                                # cv2.rectangle(annotated_frame, (int(left)-20, int(top)-20),
                                #             (int(right)+20, int(bottom)+20),
                                #             (0, 0, 255), 2) #draw rectangle to annotated_frame
                                
                                cv2.rectangle(annotated_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                                            
                                
                                print("BLINK RATIO", blinking_ratio)
                                
                                print("sau blink")
                                
                                #--padding
                                detected_face=padding(detected_face,target_size=(224,224))
                                #--end padding
                            
                                img_pixels = Image.fromarray(detected_face)
                                img_pixels = np.expand_dims(img_pixels, axis = 0)
                                samples = np.asarray(img_pixels, 'float32')
                                # prepare the face for the model, e.g. center pixels
                                    ####
                                samples = preprocess_input(samples, version=2)
                                samples_fn = np.asarray(img_pixels, 'float32')
                                samples_fn = preprocess_input(samples_fn, version=2)
                                captured_representation = vgg_model.predict(samples_fn)
                                ####
                                minratio=1
                                eucli=1000
                                
                                # try:
                                for face_id in known_persons[str(homeid[CC])]:
                                    for image_name in known_persons[str(homeid[CC])][face_id]:
                                        representation = known_persons[str(homeid[CC])][face_id][image_name] # 1D array (,150528)
                                        similarity = cosine(representation[0], captured_representation[0])
                                        # print(euclidean(known_persons[CC], captured_representation))
                                        if(similarity)<minratio: 
                                            minratio=similarity
                                            faceid=face_id
                                            imagename=image_name
                                            
                                eucli = euclidean(known_persons[str(homeid[CC])][faceid][imagename][0], captured_representation[0])
                                # except:
                                #     print(f"Nhà {homeid[CC]} chưa có ảnh nhận diện")
                                    
                                # Mở khóa nếu đúng người và đầy queue
                                # annotated_frame = frame[CC].copy()
                                if (minratio < 0.38 and eucli <90):
                                    if result_queue(q_knownperson[CC], True):
                                        lastest_detected_face[CC] = current_time
                                        print('>>> Known Person Detected ---- Minratio:',minratio, 'Eucli',eucli)
                                        
                                        cv2.putText(annotated_frame,'ID: '+str(faceid), 
                                                    (int(left)-20,int(top)-25),
                                                    cv2.FONT_HERSHEY_SIMPLEX,1 , (0, 0, 255 ), 2)

                                        
                                        # # Lưu ảnh người quen về máy
                                        # unlock_img = frame[CC].copy()
                                        
                                        # cv2.putText(unlock_img, "UNLOCK", (960,50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), thickness=3)
                                        # cv2.putText(unlock_img, f"Match img: {image_name}", (20,135), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 250), 2)
                                        # cv2.putText(unlock_img, f"Minratio:  {minratio:.5f}", (20,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 250), 2)
                                        # cv2.putText(unlock_img, f"Eucli:     {eucli:.5f}", (20,205), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 250), 2)
                                        
                                        # time_string = current_time.strftime(("%d-%m-%Y %Hh%Mm%Ss"))
                                        # cv2.imwrite(f"Unlock/{time_string}.jpg", unlock_img)
                                        
                                        # # MỞ KHÓA
                                        # access_token = get_accesstoken()
                                        # unlock(access_token=access_token, lock_id=lockid[CC])
                                        
                                        # Gửi thông báo
                                        # post_ntf_knownperson_detected(faceid, camera_name[CC], fcm_list[CC], formatted_time)
                                        # save_ntf_img(annotated_frame, 
                                        #             camera_id=get_camera_id(url[CC]), 
                                        #             title=title, 
                                        #             body=text,
                                        #             notification_type='Phát hiện người quen',
                                        #             formatted_time=formatted_time) # Lưu lại thông tin cảnh báo
                                                                
                                else:
                                    result_queue(q_knownperson[CC], False)
                                    print('Unknown Human',"min:",minratio,"eu",eucli)
                                    cv2.putText(frame[CC],'unknown human', (left-20,top-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255 ), 2)
                                        
                                    # Lưu ảnh người lạ
                                    stranger_img = frame[CC].copy()
                                    cv2.putText(stranger_img, f"Minratio:  {minratio:.3f}", (20,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 250), 2)
                                    cv2.putText(stranger_img, f"Eucli:     {eucli:.3f}", (20,205), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 250), 2)
                                    
                                    # # Lưu hình vào folder
                                    # time_string = current_time.strftime(("%d-%m-%Y %Hh%Mm%Ss"))
                                    # cv2.imwrite(f"Unknown_Human/{time_string}.jpg", stranger_img)
                                    # cv2.imwrite(f"Unknow_Human_NoBox/{time_string}.jpg", base_img)

                        #===================================== Nhận diện khung xương ===============================================#
                        
                        # Chỉ chạy nhận diện hvi khi 20s trước k có phát hiện ng quen (lastest_detected_face)
                        # Kiểm tra vị trí cả 2 tay có nằm trong vùng cần kiểm tra không
                        left_hand = keypoints_arrs[skeleton,9,:2]
                        right_hand = keypoints_arrs[skeleton,10,:2]
                                                    
                        #===========================================================================================================#
                        if lockpicking_area[CC] is not None and lastest_detected_face[CC] is None:
                            for point in lockpicking_area[CC]:
                                # Kiểm tra nếu k có thông tin vùng Mở khóa, break
                                if lockpicking_area[CC][point] is None:
                                    break
                                
                                # Nếu có thì kiểm tra 2 vị trí 2 tay với các vùng Mở khóa
                                if (inside_the_box(left_hand,lockpicking_area[CC][point]) and inside_the_box(right_hand,lockpicking_area[CC][point])):
                                    flag_inside_box_lockpicking = True
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
                                    print('Prob: ', prob, '  -   Label: ', label,  '  -   Humanconf: ', float(results[0].boxes.data[skeleton][4]*100))
                                    if prob >= pose_thres and label=='lockpicking':
                                        if result_queue(q_lockpicking[CC], True):
                                            text = 'Mở khoá' + " ({:.2f}%)".format(prob)
                                            # post_alert(fcm_list=fcm_list[CC], title=title, body=text + formatted_time_ntf)
                                            flag_save_ntf_img = True
                                            background_color = (0, 0, 255)  # Màu đỏ (B, G, R)
                                        else:
                                            text = "({:.2f}%)".format(prob)
                                            background_color = (0, 255, 0)      # Màu xanh (B, G, R)
                                    else:
                                        result_queue(q_lockpicking[CC], False)
                                        text = 'Bình thường' + " ({:.2f}%)".format(prob)
                                        background_color = (0, 255, 0)      # Màu xanh (B, G, R)
                                    print(q_lockpicking[CC].queue.count(True))
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
                                    
                                    if flag_save_ntf_img:
                                        save_ntf_img(annotated_frame, 
                                                    camera_id=get_camera_id(url[CC]), 
                                                    title=title, 
                                                    body=text,
                                                    notification_type='Pose',
                                                    formatted_time=formatted_time) # Lưu lại thông tin cảnh báo
                                        flag_save_ntf_img = False
                                # else:
                                #     result_queue(q_lockpicking[CC], False)   
                        
                        #===========================================================================================================# 
                        if climbing_area[CC] is not None and lastest_detected_face[CC] is None:
                            for point in climbing_area[CC]:
                                # Kiểm tra nếu k có thông tin vùng Mở khóa, break
                                if climbing_area[CC][point] is None:
                                    break
                                # Vẽ vùng cảnh báo
                                annotated_frame = drawbox(annotated_frame, climbing_area[CC][point])
                                # Nếu có thì kiểm tra 2 vị trí 2 tay với các vùng Leo rào
                                if (inside_the_box(left_hand,climbing_area[CC][point]) and inside_the_box(right_hand,climbing_area[CC][point])):
                                    flag_inside_box_climbing = True
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
                                    if prob >= pose_thres and label=='climbing':
                                        if result_queue(q_climbing[CC], True):
                                            text = 'Leo rào' + " ({:.2f}%)".format(prob)
                                            post_alert(fcms=fcm_list[CC], title=title, body=text + formatted_time_ntf)
                                            flag_save_ntf_img = True
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
                                    
                                    if flag_save_ntf_img==True:
                                        save_ntf_img(annotated_frame, 
                                                    camera_id=get_camera_id(url[CC]), 
                                                    title=title, 
                                                    body=text,
                                                    notification_type='Pose',
                                                    formatted_time=formatted_time)
                                    
                                else:
                                    result_queue(q_climbing[CC], False)

                if flag_nohuman:
                    result_queue(q_lockpicking[CC], False)
                    result_queue(q_climbing[CC], False)
                
                if not flag_inside_box_lockpicking:
                    result_queue(q_lockpicking[CC], False)
                if not flag_inside_box_climbing:
                    result_queue(q_climbing[CC], False)

            #-----------------------------------------------------------------------------------------------------------#
            # frame[CC] = annotated_frame.copy()
            
            # Hiện FPS
            cv2.putText(annotated_frame, "fps: {:.2f}".format(fps), (20,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 155, 255), 2)
            
            # Vẽ vùng cảnh báo
            try:
                if lockpicking_area[CC] is not None:
                    for point in lockpicking_area[CC]:
                        annotated_frame = drawbox(annotated_frame, lockpicking_area[CC][point])
                if climbing_area[CC] is not None:
                    for point in climbing_area[CC]:
                        annotated_frame = drawbox(annotated_frame, climbing_area[CC][point])
            except:
                print('Cam không có vùng cảnh báo')
                pass
            cv2.imshow(camera_name[CC], 
                        cv2.resize(annotated_frame, (int((annotated_frame.shape[1])*scale),int((annotated_frame.shape[0])*scale))))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # except Exception as e:
    #     print(e)
    #     for CC in range(len(url)):
    #         fresh[CC].release()
            
    for CC in range(len(url)):
        # fresh[CC].release()
        cap.release()
    cv2.destroyAllWindows()
    # print('Restarting...')
    
# {"L0": {"A": [650, 300], "B": [767, 300], "C": [767, 432], "D": [650, 432]}, "L1": {"A": [1240, 330], "B": [1340, 330], "C": [1340, 445], "D": [1240, 445]}}
# {"L0": {"A": [650, 300], "B": [767, 300], "C": [767, 432], "D": [650, 432]}, "L1": {"A": [1240, 330], "B": [1340, 330], "C": [1340, 445], "D": [1240, 445]}, "L2": {"A": [835, 160], "B": [1215, 160], "C": [1215, 300], "D": [835, 300]}, "L3": {"A": [760, 700], "B": [1325, 700], "C": [1415, 1079], "D": [695, 1079]}}
# {"C0": {"A": [1267, 0], "B": [1919, 0], "C": [1919, 1079], "D": [1267, 88]}, "C1": {"A": [0, 0], "B": [425, 640], "C": [257, 1079], "D": [0, 1079]}}
# pose_cls_video()
# while True:
#     try:
#         pose_cls_video()
#     except:
#         print("Lỗi nằm ngoài vòng WHILE !!!")
#         pass


while True:
    pose_cls_video()