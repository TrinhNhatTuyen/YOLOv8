import cv2, time, os, sys
sys.path.append('D:\Code\Pose\YOLOv8')
from overstepframe import FreshestFrame
url = 'rtsp://admin:Dat1qazxsw2@192.168.6.100:1554/h264_stream'
# url = 'rtsp://admin:1qazxsw2@vinaai.ddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true' # Link stream video RTSP
# url = 'rtsp://admin:NuQuynhAnh@cam24423linhdong.smartddns.tv:1554/cam/realmonitor?channel=1&subtype=0&unicast=true'

url = 'rtsp://admin:NuQuynhAnh@cam14423linhdong.smartddns.tv:1554/cam/realmonitor?channel=1&subtype=0&unicast=true' # Cam 1
# url = 'rtsp://admin:Admin123@mtkhp2408.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true'         # Cam 2
# url = 'rtsp://admin:Admin123@mtkhp2420.cameraddns.net:554/cam/realmonitor?channel=1&subtype=0&unicast=true'         # Cam 3

url = 'rtsp://admin:Vinaai!123@py1ai.cameraddns.net:5543/cam/realmonitor?channel=1&subtype=0&unicast=true'          # Cam 4
# url = 'rtsp://admin:Vinaai!123@py1ai.cameraddns.net:5541/cam/realmonitor?channel=1&subtype=0&unicast=true'          # Cam 5

# url = 'rtsp://admin:Vinaai!123@py2ai.cameraddns.net:5541/cam/realmonitor?channel=1&subtype=0&unicast=true'          # Cam 6
# url = 'rtsp://admin:Vinaai!123@py2ai.cameraddns.net:5543/cam/realmonitor?channel=1&subtype=0&unicast=true'          # Cam 7
# url = 'rtsp://admin:Vinaai!123@py2ai.cameraddns.net:5545/cam/realmonitor?channel=1&subtype=0&unicast=true'          # Cam 8
# url = 'https://admin:Vinaai!123@py2ai.cameraddns.net:80/cam/realmonitor?channel=1&subtype=0&unicast=true'           # Cam 8
# url = 'https://admin:Vinaai!123@py2ai.cameraddns.net:80'

# url = 'rtsp://admin:1qazxsw2@kbplawyer.cameraddns.net:5545/cam/realmonitor?channel=1&subtype=0&unicast=true'
# url = 'http://admin:1qazxsw2@kbplawyer.cameraddns.net:8005'
# url = 'rtsp://admin:1qazxsw2@kbplawyer.cameraddns.net:5549/cam/realmonitor?channel=1&subtype=0&unicast=true'
url = 'http://admin:1qazxsw2@kbplawyer.cameraddns.net:8010'

fresh = object()
fresh = FreshestFrame(cv2.VideoCapture(url))
frame = object()
cnt = 0

# Tạo đối tượng VideoCapture với URL stream
cap = cv2.VideoCapture(url)
scale_percent = 50
# Kiểm soát việc ghi và lưu video
record = False
stream = True
out = None
i=1
timer =time.time()
while stream:
    try:
        
        cnt, frame = fresh.read(seqnumber=cnt+1)

        base_img = frame.copy()
        
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        endtimer = time.time()
        fps = 2/(endtimer-timer)
        cv2.putText(frame, "fps: {:.2f}".format(fps), (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 155, 255), 2)
        # frame = cv2.flip(frame, 1)
        cv2.imshow('Stream', cv2.resize(frame, dim))
        
        key = cv2.waitKey(1) & 0xFF
        
        # Ấn phím 'r' để bắt đầu ghi và lưu video
        if key == ord('r') and not record:
            record = True
            # Tạo tên của video mới
            list_name = []
            for i in os.listdir('D:/Code/datatest/cam'):
                try:
                    if i.split('.')[0].isdigit():
                        list_name.append(int(i.split('.')[0]))
                except:
                    continue
            new_video_name = f'D:/Code/datatest/cam/{max(list_name)+1}.mp4'
            
            # Tạo đối tượng VideoWriter để ghi video
            out = cv2.VideoWriter(new_video_name, cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (base_img.shape[1], base_img.shape[0]))
            print('Started recording...')
        
        # Ấn phím 'q' để dừng và lưu video
        if key == ord('t') and record:
            # stream = False
            record = False
            print(f'Stopped recording. Video saved as {new_video_name}')
            out.release()
            i+=1
        
        # Ghi và lưu video nếu đang trong quá trình ghi
        if record:
            out.write(base_img)

        timer =time.time()
    except:
        continue
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
fresh.release()
cv2.destroyAllWindows()

