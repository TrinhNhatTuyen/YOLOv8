import cv2
import numpy as np
import dlib
from math import hypot
predictor_eye = dlib.shape_predictor("pre_model/shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

font = cv2.FONT_HERSHEY_PLAIN

def get_blinking_ratio(eye_points, facial_landmarks,img):
    
    # eye_points[0] - left most outer corner of the left eye 
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    
    # eye_points[3] - top-right corner of the left eye
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    
    # eye_points[1] - left most inner corner of the left eye
    # eye_points[2] - top-left corner of the left eye
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    
    # eye_points[4] - bottom-right corner of the left eye
    # eye_points[5] - bottom-left corner of the left eye
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    hor_line = cv2.line(img, left_point, right_point, (0, 255, 0), 1)
    ver_line = cv2.line(img, center_top, center_bottom, (0, 255, 0), 1)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    ratio = hor_line_lenght / ver_line_lenght
    return ratio
