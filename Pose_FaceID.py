import cv2, requests, base64, json, datetime, time, pyodbc, dlib
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from queue import Queue
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input
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

# from firebase_admin import credentials, messaging
#------------------------------------ LOAD MODEL ------------------------------------
from ultralytics import YOLO
model = YOLO('YOLOv8n-pose.pt')
pose_cls = load_model('pose_cls_v3.h5')

#------------------------------------ PARAMETERS ------------------------------------

humanpose_conf = 80
queue_len = 60
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


# face_reg_data = get_FaceRegData()

def pose_cls_video():
    download_hinhtrain()
    known_persons = load_hinhtrain()
    cam_data = get_camera_data()
    
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
    fresh, frame, cnt, first_frame, second_frame, t_oldframe, None_frame, q_knownperson, q_lockpicking, q_climbing, lastest_detected_face = [], [], [], [], [], [], [], [], [], [], []
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
        q_knownperson.append(Queue(maxsize=30))
    
    #======================= Params =======================#
    scale = 0.5
    pose_thres = 80
    thresh = 127
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
            
            # dùng để tính toán FPS
            timer =time.time()
            if t_oldframe[CC] is None:
                t_oldframe[CC] = timer
            
            # gọi lỗi nếu k đọc được frame từ camera
            if first_frame[CC] is None:
                first_frame[CC] = frame[CC]
                None_frame[CC]+=1
                continue
            
            frame[CC] = cv2.resize(frame[CC], (1920,1080))
            frame_width = frame[CC].shape[1]
            base_img = frame[CC].copy()
            prob = 0
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%d-%m-%Y %Hh%M'%S\"")
            
            fps = 1/(timer-t_oldframe[CC])
            t_oldframe[CC] = timer
            
            first_frame[CC] = cv2.cvtColor(first_frame[CC], cv2.COLOR_BGR2GRAY)
            first_frame[CC] = cv2.threshold(first_frame[CC], thresh, 255, cv2.THRESH_BINARY)[1]

            second_frame[CC] = frame[CC]
            second_frame[CC] = cv2.cvtColor(second_frame[CC], cv2.COLOR_BGR2GRAY)
            second_frame[CC] = cv2.threshold(second_frame[CC], thresh, 255, cv2.THRESH_BINARY)[1]

            res = cv2.absdiff(first_frame[CC], second_frame[CC])
            first_frame[CC] = frame[CC]
            res = res.astype(np.uint8)
            percentage = (np.count_nonzero(res) * 100)/ res.size
            # print('FRAM DIFF: ', percentage)

            original_size = base_img.shape
            target_size = (300, 300)
            # img = cv2.resize(frame, target_size)
            aspect_ratio_x = (original_size[1] / target_size[1])
            aspect_ratio_y = (original_size[0] / target_size[0])
            
            #====================================== Nhận diện khuôn mặt ================================================#
            if percentage>=0.1 and task[CC]=='FaceID':
                # Call Face Detect
                detections_df = detect_face(frame[CC],detector)
                # Nếu không có ai trong khung hình
                if len(detections_df)==0:
                    # Lưu lại thời gian phát hiện khuôn mặt ng quen
                    # Trong 20s camera hvi đi cùng với cam FaceID hiện tại sẽ k chạy nhận diện
                    
                    index_of_pose_camera = camera_id.index(related_camera_id[CC])
                    if lastest_detected_face[index_of_pose_camera] is not None:
                        time_difference = current_time - lastest_detected_face[index_of_pose_camera]
                        if time_difference.total_seconds() >= 20:
                            lastest_detected_face[index_of_pose_camera] = None
                else:
                    for _, instance in detections_df.iterrows():
                        # confidence_score = str(round(100*instance["confidence"], 2))+" %"
                        
                        # Cắt ảnh khuôn mặt người để so khớp
                        left = instance["left"]; right = instance["right"]
                        bottom = instance["bottom"]; top = instance["top"]
                        detected_face = base_img[int(top*aspect_ratio_y):int(bottom*aspect_ratio_y),
                                                int(left*aspect_ratio_x):int(right*aspect_ratio_x)]
                        
                        # Cắt ảnh khuôn mặt người để lưu lại
                        # detected_face1 = base_img[int(top*aspect_ratio_y)-20:int(bottom*aspect_ratio_y)+20,
                        #                         int(left*aspect_ratio_x)-20:int(right*aspect_ratio_x)+20]
                        top1 = int(top*aspect_ratio_y)-20 if (int(top*aspect_ratio_y)-20)>=0 else 0
                        bottom1 = int(bottom*aspect_ratio_y)+20 if (int(top*aspect_ratio_y)+20)<original_size[1] else original_size[1]
                        left1 = int(left*aspect_ratio_x)-20 if (int(left*aspect_ratio_x)-20)>=0 else int(left*aspect_ratio_x)-20
                        right1 = int(right*aspect_ratio_x)+20 if (int(right*aspect_ratio_x)+20)<original_size[0] else original_size[0]
                        detected_face1 = base_img[top1:bottom1, left1:right1]
                    
                        ## lanmark face eyes
                        gey = dlib.rectangle(int(left*aspect_ratio_x), int(top*aspect_ratio_y), 
                                            int(right*aspect_ratio_x), int(bottom*aspect_ratio_y))
                        landmarks = predictor_eye(frame[CC], gey)
                        # Get eyes positions
                        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], landmarks,frame[CC])
                        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], landmarks,frame[CC])
                        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
                        cv2.rectangle(frame[CC], (int(left*aspect_ratio_x)-20, int(top*aspect_ratio_y)-20),
                                    (int(right*aspect_ratio_x)+20, int(bottom*aspect_ratio_y)+20),
                                    (0, 0, 255), 1) #draw rectangle to main image
                        ##
                        print("BLINK RATIO", blinking_ratio)
                        
                        print("sau blink")
                        cv2.rectangle(frame[CC], (int(left*aspect_ratio_x)-20, int(top*aspect_ratio_y)-20), (int(right*aspect_ratio_x)+20, int(bottom*aspect_ratio_y)+20), (0, 255, 0), 2) #draw rectangle to main image
                        detected_face = frame[CC][int(top*aspect_ratio_y):int(bottom*aspect_ratio_y), int(left*aspect_ratio_x):int(right*aspect_ratio_x)] #crop detected face
                        
                        # #--padding
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
                        captured_representation = model.predict(samples_fn)
                        ####
                        minratio=1
                        # faceid='noname'
                        for face_id in known_persons[str(homeid[CC])]:
                            for image_name in known_persons[str(homeid[CC])][face_id]:
                                representation = known_persons[str(homeid[CC])][face_id][image_name] # 1D array (,150528)
                                similarity = cosine(representation, captured_representation[0])
                                # print(euclidean(known_persons[CC], captured_representation))
                                if(similarity)<minratio: 
                                    minratio=similarity
                                    faceid=face_id
                                    imagename=image_name
                                    
                        eucli = euclidean(known_persons[str(homeid[CC])][faceid][imagename], captured_representation[0])
                        # Mở khóa nếu đúng người và đầy queue
                        if (minratio < 0.38 and eucli <90):
                            if result_queue(q_knownperson[CC], True):
                                lastest_detected_face[CC] = current_time
                                print('>>> Known Person Detected ---- Minratio:',minratio, 'Eucli',eucli)
                                
                                cv2.putText(frame[CC],'ID: '+str(faceid).split('_')[1] , 
                                            (int(left*aspect_ratio_x)-20,int(top*aspect_ratio_y)-25),
                                            cv2.FONT_HERSHEY_SIMPLEX,1 , (0, 0,255 ), 2)

                                
                                # Lưu ảnh người quen về máy
                                unlock_img = frame[CC].copy()
                                cv2.putText(unlock_img, "UNLOCK", (960,50), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,0), thickness=3)
                                cv2.putText(unlock_img, f"Match img: {image_name}", (20,135), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 250), 2)
                                cv2.putText(unlock_img, f"Minratio:  {minratio:.5f}", (20,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 250), 2)
                                cv2.putText(unlock_img, f"Eucli:     {eucli:.5f}", (20,205), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 250), 2)
                                
                                time_string = current_time.strftime(("%d-%m-%Y %Hh%Mm%Ss"))
                                cv2.imwrite(f"Unlock/{time_string}.jpg", unlock_img)
                                
                                # MỞ KHÓA
                                access_token = get_accesstoken()
                                unlock(access_token=access_token, lock_id=lockid[CC])
                                
                                # Gửi thông báo
                                post_ntf_knownperson_detected(faceid, camera_name[CC], fcm_list[CC], formatted_time)
                                save_ntf_img(frame[CC], 
                                            camera_id=get_camera_id(url[CC]), 
                                            title=title, 
                                            body=text,
                                            notification_type='Phát hiện người quen',
                                            formatted_time=formatted_time) # Lưu lại thông tin cảnh báo
                                                        
                        else:
                            result_queue(q_knownperson[CC], False)
                            print('Unknown Human',"min:",minratio,"eu",eucli)
                            cv2.putText(frame[CC],'unknown human', (int(left*aspect_ratio_x)-20,int(top*aspect_ratio_y)-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,255 ), 2)
                                
                            # Lưu ảnh người lạ
                            stranger_img = frame[CC].copy()
                            cv2.putText(stranger_img, f"Minratio:  {minratio:.5f}", (20,170), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 250), 2)
                            cv2.putText(stranger_img, f"Eucli:     {eucli:.5f}", (20,205), cv2.FONT_HERSHEY_SIMPLEX, 1, (102, 255, 250), 2)
                            
                            time_string = current_time.strftime(("%d-%m-%Y %Hh%Mm%Ss"))
                            cv2.imwrite(f"Unknown_Human/{time_string}.jpg", stranger_img)
                            cv2.imwrite(f"Unknow_Human_NoBox/{time_string}.jpg", base_img)
            
            
            #===================================== Nhận diện khung xương ===============================================#
            if percentage>=0.1 and task[CC]=='Pose' and (lastest_detected_face[CC] is None):
                results = model(frame[CC], save=False)
                annotated_frame = results[0].plot(boxes=False)
                
                keypoints_arrs = results[0].keypoints.data.numpy()
                #-----------------------------------------------------------------------------------------------------------#      
                # Nếu không phát hiện người
                if len(results[0].boxes.data)==0: 
                    result_queue(q_lockpicking[CC], False)
                    result_queue(q_climbing[CC], False)
                # Nếu phát hiện người
                else:
                    # Lặp qua từng khung xương:
                    for skeleton in range(len(keypoints_arrs)):
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
                                        if prob >= pose_thres and label=='lockpicking':
                                            if result_queue(q_lockpicking[CC], True):
                                                text = 'Mở khoá' + " ({:.2f}%)".format(prob)
                                                post_alert(fcm_list=fcm_list[CC], title=title, body=text)
                                                save_ntf_img(annotated_frame, 
                                                            camera_id=get_camera_id(url[CC]), 
                                                            title=title, 
                                                            body=text,
                                                            notification_type='Cảnh báo',
                                                            formatted_time=formatted_time) # Lưu lại thông tin cảnh báo
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
                                        if prob >= pose_thres and label=='climbing':
                                            if result_queue(q_climbing[CC], True):
                                                text = 'Leo rào' + " ({:.2f}%)".format(prob)
                                                post_alert(fcms=fcm_list[CC], title=title, body=text)
                                                save_ntf_img(annotated_frame, 
                                                            camera_id=get_camera_id(url[CC]), 
                                                            title=title, 
                                                            body=text,
                                                            notification_type='Cảnh báo',
                                                            formatted_time=formatted_time)
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
            #-----------------------------------------------------------------------------------------------------------#
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
    

# a = get_camera_data()  
pose_cls_video()
# while True:
#     try:
#         pose_cls_video()
#     except:
#         print("Lỗi nằm ngoài vòng WHILE !!!")
#         pass
