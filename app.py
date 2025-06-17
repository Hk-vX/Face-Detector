import cv2
import streamlit as st
import numpy as np

st.set_page_config(page_title="Face Detection", layout="centered")
st.title("Face Detection")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
    st.markdown(f"### Faces Detected: `{len(faces)}`")
else:
    st.info("Upload an image to begin face detection.")
