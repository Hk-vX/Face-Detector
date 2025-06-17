import cv2
import streamlit as st
import numpy as np

st.set_page_config(page_title="Face Detection", layout="centered")

st.title("Real-Time Face Detection")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

start_button = st.button("Start Camera")
stop_button = st.button("Stop Camera")

frame_placeholder = st.empty()
face_count_placeholder = st.empty()

if 'run' not in st.session_state:
    st.session_state.run = False

if start_button:
    st.session_state.run = True

if stop_button:
    st.session_state.run = False

if st.session_state.run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            face_count = len(faces)
            face_count_placeholder.markdown(f"### 🧑‍🤝‍🧑 Faces Detected: {face_count}")

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB")

        cap.release()
        frame_placeholder.empty()
        face_count_placeholder.empty()
else:
    st.info("Click **Start Camera** to begin.")
