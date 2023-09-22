import cv2, requests, base64, json, datetime, time, pyodbc
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from queue import Queue
from tensorflow.keras.models import load_model
from overstepframe import FreshestFrame
from inside_the_box import inside_the_box
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

def post_alert(fcm_list, title, body, data=None):
    for fcm in fcm_list:
        # Đường dẫn API FCM
        url = 'https://fcm.googleapis.com/fcm/send'
        
        # Đặt thông báo đẩy
        payload = {
            'to': fcm,
            'notification': {
                'title': title,
                'body': body
            },
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

def get_FaceRegData():
    connection = pyodbc.connect("Driver={SQL Server};"
                                "Server=112.78.15.3;"
                                "Database=VinaAIAPP;"
                                "uid=ngoi;"
                                "pwd=admin123;")
    cursor = connection.cursor()

    try:
        cursor.execute("SELECT * FROM FaceRegData")
        rows = cursor.fetchall()
        
        result = []
        for row in rows:
            data_dict = {
                "ImageID": row.ImageID,
                "FaceID": row.FaceID,
                "FaceName": row.FaceName,
                "HomeID": row.HomeID,
                "ImagePath": row.ImagePath,
                "ImageArray": base64_to_array(row.Base64)
            }
            result.append(data_dict)
        
        return result
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        cursor.close()
        connection.close()

def update_data(data):
    """ Cập nhật lại list data
    Args:
        data (list): list thông tin nhận diện, trả về từ hàm "get_FaceRegData" được gọi lúc ban đầu
    """
    connection = pyodbc.connect("Driver={SQL Server};"
                                "Server=112.78.15.3;"
                                "Database=VinaAIAPP;"
                                "uid=ngoi;"
                                "pwd=admin123;")
    cursor = connection.cursor()
    
    cursor.execute("SELECT ImageID FROM FaceRegData")
    rows = cursor.fetchall()
    for row in rows:
        image_id = row.ImageID
        # Kiểm tra xem ImageID có trong danh sách data không
        image_id_exists = any(entry["ImageID"] == image_id for entry in data)
        if not image_id_exists:
            try:
                cursor.execute("SELECT * FROM FaceRegData WHERE ImageID=?", (image_id,))
                row = cursor.fetchone()
                if row:
                    data_dict = {
                        "ImageID": row.ImageID,
                        "FaceID": row.FaceID,
                        "FaceName": row.FaceName,
                        "HomeID": row.HomeID,
                        "ImagePath": row.ImagePath,
                        "ImageArray": base64_to_array(row.Base64)
                    }
                    data.append(data_dict)
            except Exception as e:
                print(f"Error: {str(e)}")
            finally:
                cursor.close()
                connection.close()


# face_reg_data = get_FaceRegData()

def pose_cls_video():
    # global face_reg_data
    cam_data = get_camera_data()
    
    #------------------------------------ Các thông tin của Camera ------------------------------------
    url, lockpicking_area, climbing_area, fcm_list, camera_id, camera_name, homeid, lockid, related_camera_id = [], [], [], [], [], [], [], [], []
    # Bỏ qua các cam chưa nhập LockpickingArea & ClimbingArea và không có LockID
    for cam in cam_data:
        if (cam['LockpickingArea'] is None) and (cam['ClimbingArea'] is None) and (cam['LockID'] is None):
            url.append(cam['RTSP'])
            lockpicking_area.append(cam['LockpickingArea'])
            climbing_area.append(cam['ClimbingArea'])
            fcm_list.append(cam['FCM'])
            camera_id.append(cam['CameraID'])
            related_camera_id.append(cam['RelatedCameraID'])
            camera_name.append(cam['CameraName'])
            homeid.append(cam['HomeID'])
            lockid.append(cam['LockID'])
            
    #------------------------------------ FRESHEST FRAME ------------------------------------
    fresh, frame, cnt, first_frame, second_frame, t_oldframe, None_frame, q_lockpicking, q_climbing = [], [], [], [], [], [], [], [], []
    for i in range(len(url)):
        fresh.append(FreshestFrame(cv2.VideoCapture(url[i])))
        frame.append(object())
        cnt.append(0)
        first_frame.append(None)
        second_frame.append(None)
        t_oldframe.append(None)
        None_frame.append(0)
        
        q_lockpicking.append(Queue(maxsize=queue_len))
        q_climbing.append(Queue(maxsize=queue_len))
        
    scale = 0.5
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
                                    if prob >= thres and label=='lockpicking':
                                        if result_queue(q_lockpicking[CC], True):
                                            text = 'Mở khoá' + " ({:.2f}%)".format(prob)
                                            post_alert(fcm_list=fcm_list[CC], title=title, body=text)
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
                                    
                        
                        #===========================================================================================================# 
                        if lockpicking_area[CC] is not None:
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
    

a = get_camera_data()  
pose_cls_video()
# while True:
#     try:
#         pose_cls_video()
#     except:
#         print("Lỗi nằm ngoài vòng WHILE !!!")
#         pass
