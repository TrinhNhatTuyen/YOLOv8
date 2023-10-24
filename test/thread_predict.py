import threading, time
from ultralytics import YOLO

def predict_yolo(yolo_model, image_path):
    result = yolo_model.predict(image_path)
    # print(result)  # In kết quả dự đoán

image_path = "test/hinh/8.jpg"

model1 = YOLO('YOLOv8n-pose.pt')
model2 = YOLO('FireSmokePerson_v2.pt')
predict_yolo(model1, image_path)
predict_yolo(model2, image_path)
#----------------------------Không chia luồng-----------------------------
start_time = time.time()

image_path = "test/hinh/4.jpg"

predict_yolo(model1, image_path)
predict_yolo(model2, image_path)

elapsed_time2 = time.time() - start_time


#----------------------------Chia luồng-----------------------------
thread1 = threading.Thread(target=predict_yolo, args=(model1, image_path))
thread2 = threading.Thread(target=predict_yolo, args=(model2, image_path))

start_time = time.time()

thread1.start()
thread2.start()
start_time = time.time()
thread1.join()
thread2.join()

elapsed_time1 = time.time() - start_time


#--------------------------------------------------------------------
print(f"Không chia luồng: {elapsed_time2} giây")
print(f"Có chia luồng: {elapsed_time1} giây")