import os
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')
save_dir = "D:/Code/Pose/YOLOv8/runs/pose/predict6"
# source_directories = ["D:/Code/datatest/lockpicking", "D:/Code/datatest/breaking", "D:/Code/datatest/climbing"]
# des = ["D:/Code/Pose/YOLOv8/lockpicking", "D:/Code/Pose/YOLOv8/breaking", "D:/Code/Pose/YOLOv8/climbing"]

source_directories = ["D:/Code/FaceID/Unknow_Human_NoBox"]
des = ["D:/Code/Pose/YOLOv8/Unknow_Human_Pose"]

for dir in des:
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        os.remove(file_path)
            
# Dự đoán trên từng ảnh trong các thư mục
for directory in source_directories:
    for file_name in os.listdir(directory):
        try:
            source = os.path.join(directory, file_name)
            model.predict(source, save=True, imgsz=320, conf=0.5, classes=[0])

            # Di chuyển ảnh kết quả vào các thư mục tương ứng
            # if directory.endswith("lockpicking"):
            #     target_dir = "D:/Code/Pose/YOLOv8/lockpicking"
            # elif directory.endswith("breaking"):
            #     target_dir = "D:/Code/Pose/YOLOv8/breaking"
            # elif directory.endswith("climbing"):
            #     target_dir = "D:/Code/Pose/YOLOv8/climbing"
            
            target_dir = des[0]

            os.makedirs(target_dir, exist_ok=True)
            target_file = os.path.join(target_dir, file_name)
            os.rename(os.path.join(save_dir, file_name), target_file)
            
        except Exception as e:
            print('ERROR: ',e)
