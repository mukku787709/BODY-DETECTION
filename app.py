import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import cv2
import mediapipe as mp
import numpy as np
import av
from utils import eye_aspect_ratio, mouth_aspect_ratio, head_pose_estimation, log_event, play_alert_sound
import time
import json
import os

# Initialize MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Thresholds
EAR_THRESH = 0.25
EAR_CONSEC_FRAMES = 10
MAR_THRESH = 0.15
MAR_CONSEC_FRAMES = 30
HEAD_TILT_THRESH = 25

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

class VideoProcessor:
    def __init__(self):
        self.ear_counter = 0
        self.mar_counter = 0
        self.alert_active = False
        self.face_mesh = mp_face_mesh.FaceMesh(max_num_faces=10, refine_landmarks=True)
        self.hands = mp_hands.Hands(max_num_hands=20)

    async def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = cv2.flip(img, 1)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process face
        face_results = self.face_mesh.process(rgb_img)
        # Process hands
        hand_results = self.hands.process(rgb_img)

        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                landmarks = np.array([(lm.x * img.shape[1], lm.y * img.shape[0]) for lm in face_landmarks.landmark])

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
                head_tilt = head_pose_estimation(landmarks, img.shape)

                # Draw face mesh
                mp_drawing.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                # Check drowsiness
                if ear < EAR_THRESH:
                    self.ear_counter += 1
                    if self.ear_counter >= EAR_CONSEC_FRAMES:
                        cv2.putText(img, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        if not self.alert_active:
                            play_alert_sound()
                            log_event("drowsiness", {"ear": ear})
                            self.alert_active = True
                else:
                    self.ear_counter = 0
                    self.alert_active = False

                # Check yawning
                if mar > MAR_THRESH:
                    self.mar_counter += 1
                    if self.mar_counter >= MAR_CONSEC_FRAMES:
                        cv2.putText(img, "YAWNING ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        play_alert_sound()
                        log_event("yawning", {"mar": mar})
                        self.mar_counter = 0
                else:
                    self.mar_counter = 0

                # Check head pose
                if head_tilt > HEAD_TILT_THRESH:
                    cv2.putText(img, "BAD POSTURE ALERT!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    play_alert_sound()
                    log_event("bad_posture", {"tilt": head_tilt})

        # Process hands
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers_curled = True  # Placeholder
                if fingers_curled:
                    cv2.putText(img, "OBJECT IN HAND", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    log_event("object_in_hand", {})

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def load_event_counts():
    log_file = 'data/events.json'
    if not os.path.exists(log_file):
        return {"drowsiness": 0, "yawning": 0, "bad_posture": 0, "object_in_hand": 0}
    counts = {"drowsiness": 0, "yawning": 0, "bad_posture": 0, "object_in_hand": 0}
    with open(log_file, 'r') as f:
        for line in f:
            try:
                event = json.loads(line.strip())
                event_type = event.get('type')
                if event_type in counts:
                    counts[event_type] += 1
            except json.JSONDecodeError:
                pass
    return counts

def main():
    st.set_page_config(
        page_title="Body Detection System",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Enhanced Dark Theme CSS
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1e1e2e, #2a2d3a);
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .css-1d391kg {
        background: linear-gradient(135deg, #1e1e2e, #2a2d3a);
    }
    .css-1v3fvcr {
        background-color: #3a3f4b;
        color: #e0e0e0;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
    }
    .stSidebar {
        background: linear-gradient(180deg, #2a2d3a, #1e1e2e);
        border-right: 2px solid #4a4f5a;
    }
    .stMetric {
        background: linear-gradient(135deg, #4a4f5a, #3a3f4b);
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.3);
        color: #e0e0e0;
    }
    .stTitle {
        color: #ffffff;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    .stSubheader {
        color: #b0b0b0;
        font-weight: 600;
    }
    .stMarkdown {
        color: #d0d0d0;
    }
    .css-1v0mbdj {
        background-color: #4a4f5a;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("👁️ Advanced Body Detection System")
    st.markdown("**Real-time AI-powered monitoring** for drowsiness, yawning, posture, and hand object detection using your webcam.")

    # Sidebar with enhanced stats
    st.sidebar.title("📊 Real-Time Analytics")
    counts = load_event_counts()
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("😴 Drowsiness", counts["drowsiness"], delta="alerts")
        st.metric("😮 Yawning", counts["yawning"], delta="alerts")
    with col2:
        st.metric("🧍 Bad Posture", counts["bad_posture"], delta="alerts")
        st.metric("🤏 Object in Hand", counts["object_in_hand"], delta="alerts")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Detection Parameters:**")
    st.sidebar.write(f"• Eye Aspect Ratio: {EAR_THRESH}")
    st.sidebar.write(f"• Mouth Aspect Ratio: {MAR_THRESH}")
    st.sidebar.write(f"• Head Tilt Threshold: {HEAD_TILT_THRESH}°")

    # Main content with better layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎥 Live Video Stream")
        st.markdown("Click 'START' to begin real-time detection.")
        webrtc_ctx = webrtc_streamer(
            key="body-detection",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
        )

    with col2:
        st.subheader("📋 How It Works")
        st.markdown("""
        **1. Grant Camera Access**  
        Allow browser to access your webcam.

        **2. Position Yourself**  
        Face the camera clearly for accurate detection.

        **3. Monitor Alerts**  
        Visual alerts appear on video for detected issues.

        **4. View Statistics**  
        Check sidebar for cumulative event counts.

        **Features:**  
        - 🚨 Drowsiness detection via eye tracking  
        - 😮 Yawning detection via mouth analysis  
        - 🧍 Posture monitoring via head tilt  
        - 🤏 Hand object placeholder detection
        """)

        st.markdown("---")
        st.markdown("**System Status:** Active ✅")

if __name__ == "__main__":
    main()
