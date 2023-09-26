import cv2, pickle, os
from ultralytics import YOLO
import numpy as np

# Load a model
model = YOLO('YOLOv8n-pose.pt')

############ ------------>>> https://docs.ultralytics.com/modes/predict/#arguments <<<------------ ############

list_arr = []

import cv2, os
from ultralytics import YOLO
model = YOLO('YOLOv8n-pose.pt')
video_type = 'smthgelse'
# video_type = 'climbing'
# video_type = 'lockpicking'
folder_path = f'D:/Code/datatest/{video_type}'
for file_name in os.listdir(folder_path):
    file_path = folder_path + '/' + file_name
    if os.path.isfile(file_path) and file_name.endswith('.mp4'):

        pkl_file = f'{video_type}/' + file_path.split('/')[-1].split('.')[0] + '.pkl'
        if not os.path.exists(pkl_file):
        
            cap = cv2.VideoCapture(file_path)

            # Xác định thông số video đầu vào
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Tạo đối tượng VideoWriter để ghi video
            output_video = f'{video_type}/' + file_path.split('/')[-1].split('.')[0] + '.mp4'
            out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (frame_width, frame_height))

            while cap.isOpened():
                ret, frame = cap.read()
                if ret:

                    
                    results = model(frame, save=False)
                    flip_results = model(cv2.flip(frame, 1), save=False)
                    
                    annotated_frame = results[0].plot()
                    
                    out.write(annotated_frame)
                    
                    # cv2.putText(annotated_frame, "fps: {:.2f}".format(fps), (20,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (23, 155, 255), 2)
                    frame = cv2.resize(annotated_frame, ((int)((annotated_frame.shape[1])*0.6),(int)((annotated_frame.shape[0])*0.6)))
                    cv2.imshow(file_name, frame)
                    
                    keypoints_arrs = results[0].keypoints.data.numpy()
                    flip_keypoints_arrs = flip_results[0].keypoints.data.numpy()
            # # ------------------------------------------------------------------------ #
                    try:
                        # Nếu conf của object số 0 được lớn hơn 0.75
                        if float(results[0].boxes.data[0][4]) >= 0.75:
                            
                            keypoints_arrs[0,:,0] = keypoints_arrs[0,:,0]/frame_width
                            keypoints_arrs[0,:,1] = keypoints_arrs[0,:,1]/frame_width
                            x_min = np.min(keypoints_arrs[0,:,0])
                            y_min = np.min(keypoints_arrs[0,:,1])
                            keypoints_arrs[0,:,0] -= x_min
                            keypoints_arrs[0,:,1] -= y_min
                            list_arr.append(keypoints_arrs[0,:,:])
                            
                            flip_keypoints_arrs[0,:,0] = flip_keypoints_arrs[0,:,0]/frame_width
                            flip_keypoints_arrs[0,:,1] = flip_keypoints_arrs[0,:,1]/frame_width
                            flip_x_min = np.min(flip_keypoints_arrs[0,:,0])
                            flip_y_min = np.min(flip_keypoints_arrs[0,:,1])
                            flip_keypoints_arrs[0,:,0] -= flip_x_min
                            flip_keypoints_arrs[0,:,1] -= flip_y_min
                            list_arr.append(flip_keypoints_arrs[0,:,:])
                    except:
                        continue
                    

                    try:
                        if file_name.startswith("Toankungfu"):
                            if len(keypoints_arrs[1])==17:
                                
                                keypoints_arrs[1,:,0] = keypoints_arrs[1,:,0]/frame_width
                                keypoints_arrs[1,:,1] = keypoints_arrs[1,:,1]/frame_width
                                x_min = np.min(keypoints_arrs[1,:,0])
                                y_min = np.min(keypoints_arrs[1,:,1])
                                keypoints_arrs[1,:,0] -= x_min
                                keypoints_arrs[1,:,1] -= y_min
                                list_arr.append(keypoints_arrs[1,:,:])
                                
                                flip_keypoints_arrs[1,:,0] = flip_keypoints_arrs[1,:,0]/frame_width
                                flip_keypoints_arrs[1,:,1] = flip_keypoints_arrs[1,:,1]/frame_width
                                flip_x_min = np.min(flip_keypoints_arrs[1,:,0])
                                flip_y_min = np.min(flip_keypoints_arrs[1,:,1])
                                flip_keypoints_arrs[1,:,0] -= flip_x_min
                                flip_keypoints_arrs[1,:,1] -= flip_y_min
                                list_arr.append(flip_keypoints_arrs[1,:,:])
                    except:
                        continue
                else:
                    break
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            
            with open(pkl_file, 'wb') as file:
                pickle.dump(np.array(list_arr), file)

# with open(pkl_file, 'rb') as file:
#     loaded_data = pickle.load(file)
#     print()

