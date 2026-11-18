import cv2
import numpy as np
import gradio as gr
from PIL import Image

def face_detection(image):
    # convert the imput image to a numpy array
    image = np.array(image)
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # load the pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # detect face in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeightbors=5)
    # draw rectangles around the detected faces
    for (x,y,w,h) in faces:
        cv2.rectangle(image, (x,y),(x+w,y+h), (255,0,0), 2)

    return image

# create a Gradio interface for the face detection function
iface = gr.Interface(
    fn=face_detection,
    inputs = "image",
    outputs = "image",
    title = "face fetection",
    description = "Upload an image and the app will detect faces in it using OpenCV's Haar Cascade classifier."
    )
iface.launch()
    