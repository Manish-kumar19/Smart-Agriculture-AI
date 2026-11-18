
import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Haar Face Detection")
st.write("Upload an image to detect faces using OpenCV.")

def face_detection(image):
    image = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(
            image,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    return image, len(faces)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    result, count = face_detection(image)

    st.image(result, caption="Face Detection Result")
    st.success(f"Detected {count} face(s).")
