import cv2
import mediapipe as mp
import numpy as np
from utils import eye_aspect_ratio, mouth_aspect_ratio, head_pose_estimation, log_event, play_alert_sound
import time

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Thresholds
EAR_THRESH = 0.25
EAR_CONSEC_FRAMES = 10
MAR_THRESH = 0.15  # Mouth opening percentage
MAR_CONSEC_FRAMES = 30  # Assuming 30 FPS, 1 second
HEAD_TILT_THRESH = 25

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera, trying index 1")
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            print("Cannot open camera. Please ensure camera is connected and permissions granted.")
            return

    # Counters
    ear_counter = 0
    mar_counter = 0
    alert_active = False

    with mp_face_mesh.FaceMesh(max_num_faces=10, refine_landmarks=True) as face_mesh, \
         mp_hands.Hands(max_num_hands=20) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process face
            face_results = face_mesh.process(rgb_frame)
            # Process hands
            hand_results = hands.process(rgb_frame)

            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # Get landmarks
                    landmarks = np.array([(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark])

                    # Eye aspect ratio
                    left_eye = landmarks[[33, 160, 158, 133, 153, 144]]
                    right_eye = landmarks[[362, 385, 387, 263, 373, 380]]
                    left_ear = eye_aspect_ratio(left_eye)
                    right_ear = eye_aspect_ratio(right_eye)
                    ear = (left_ear + right_ear) / 2.0

                    # Mouth aspect ratio
                    mouth = landmarks[[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88]]
                    mar = mouth_aspect_ratio(mouth)

                    # Head pose
                    head_tilt = head_pose_estimation(landmarks, frame.shape)

                    # Draw face mesh
                    mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                    # Check drowsiness
                    if ear < EAR_THRESH:
                        ear_counter += 1
                        if ear_counter >= EAR_CONSEC_FRAMES:
                            cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            if not alert_active:
                                play_alert_sound()
                                log_event("drowsiness", {"ear": ear})
                                alert_active = True
                    else:
                        ear_counter = 0
                        alert_active = False

                    # Check yawning
                    if mar > MAR_THRESH:
                        mar_counter += 1
                        if mar_counter >= MAR_CONSEC_FRAMES:
                            cv2.putText(frame, "YAWNING ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            play_alert_sound()
                            log_event("yawning", {"mar": mar})
                            mar_counter = 0  # Reset after alert
                    else:
                        mar_counter = 0

                    # Check head pose
                    if head_tilt > HEAD_TILT_THRESH:
                        cv2.putText(frame, "BAD POSTURE ALERT!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        play_alert_sound()
                        log_event("bad_posture", {"tilt": head_tilt})

            # Process hands
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # Simplified object detection: check if fingers are curled
                    # This is a basic check; real implementation would be more complex
                    fingers_curled = True  # Placeholder
                    if fingers_curled:
                        cv2.putText(frame, "OBJECT IN HAND", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        log_event("object_in_hand", {})

            cv2.imshow('Body Detection System', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
