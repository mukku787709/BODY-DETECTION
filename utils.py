import numpy as np
import cv2
import json
import os
from datetime import datetime

def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])
    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    # Vertical distances
    A = np.linalg.norm(mouth[13] - mouth[19])  # Upper lip to lower lip
    B = np.linalg.norm(mouth[14] - mouth[18])
    C = np.linalg.norm(mouth[15] - mouth[17])
    # Horizontal distance
    D = np.linalg.norm(mouth[12] - mouth[16])
    # Mouth aspect ratio
    mar = (A + B + C) / (3.0 * D)
    return mar

def head_pose_estimation(face_landmarks, image_shape):
    # Simplified head pose using nose and eyes
    nose = face_landmarks[1]
    left_eye = face_landmarks[33]
    right_eye = face_landmarks[263]

    # Calculate tilt angle
    eye_center = (left_eye + right_eye) / 2
    dx = nose[0] - eye_center[0]
    dy = nose[1] - eye_center[1]
    angle = np.arctan2(dy, dx) * 180 / np.pi
    return abs(angle)

def log_event(event_type, details):
    log_dir = 'data'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'events.json')
    event = {
        'timestamp': datetime.now().isoformat(),
        'type': event_type,
        'details': details
    }
    with open(log_file, 'a') as f:
        json.dump(event, f)
        f.write('\n')

def play_alert_sound():
    # For simplicity, we'll use a beep; in real implementation, use playsound
    import winsound
    winsound.Beep(800, 500)  # 800Hz for 500ms
