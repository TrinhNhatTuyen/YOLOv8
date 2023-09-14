import cv2, time, os, sys
sys.path.append('D:\Code\Pose\YOLOv8')
from overstepframe import FreshestFrame
# url = 'rtsp://admin:Dat1qazxsw2@192.168.6.100:1554/h264_stream'
# url = 'rtsp://admin:1qazxsw2@vinaai.ddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true' # Link stream video RTSP
# url = 'rtsp://admin:NuQuynhAnh@cam24423linhdong.smartddns.tv:1554/cam/realmonitor?channel=1&subtype=0&unicast=true'

# url = 'rtsp://admin:NuQuynhAnh@cam14423linhdong.smartddns.tv:1554/cam/realmonitor?channel=1&subtype=0&unicast=true' # Cam 1
# url = 'rtsp://admin:Admin123@mtkhp2408.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true'         # Cam 2
# url = 'rtsp://admin:Admin123@mtkhp2420.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true'         # Cam 3

url = [
        # 'rtsp://admin:NuQuynhAnh@cam14423linhdong.smartddns.tv:1554/cam/realmonitor?channel=1&subtype=0&unicast=true', # Cam 1
        # 'rtsp://admin:Admin123@mtkhp2408.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true'         # Cam 2
        # 'rtsp://admin:Admin123@mtkhp2420.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true'         # Cam 3
        # 'rtsp://admin:Vinaai!123@py1ai.cameraddns.net:5543/cam/realmonitor?channel=1&subtype=0&unicast=true',           # Cam 4
        'rtsp://admin:Vinaai!123@py1ai.cameraddns.net:5541/cam/realmonitor?channel=1&subtype=0&unicast=true',           # Cam 5
        # 'rtsp://admin:Vinaai!123@py2ai.cameraddns.net:5541/cam/realmonitor?channel=1&subtype=0&unicast=true',           # Cam 6
        # 'rtsp://admin:Vinaai!123@py2ai.cameraddns.net:5543/cam/realmonitor?channel=1&subtype=0&unicast=true',           # Cam 7
        # 'rtsp://admin:Vinaai!123@py2ai.cameraddns.net:5545/cam/realmonitor?channel=1&subtype=0&unicast=true',           # Cam 8
        ]

fresh, frame, cnt, first_frame, second_frame, None_frame, cam_name = [], [], [], [], [], [], []
for i in range(len(url)):
    fresh.append(object())
    frame.append(object())
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
            
            frame[CC] = cv2.resize(frame[CC], (1920,1080))
            scale_percent = 50 # percent of original size
            width = int(frame[CC].shape[1] * scale_percent / 100)
            height = int(frame[CC].shape[0] * scale_percent / 100)
            dim = (width, height)
            # cv2.imwrite('test/Cam8.jpg', cv2.resize(frame[CC], (1920,1080)))
            cv2.imshow(cam_name[CC], cv2.resize(frame[CC], dim))
            print(CC)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
except:
    pass

#-------------------------------------------------------------------------------------------------------------

