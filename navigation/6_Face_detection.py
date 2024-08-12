import streamlit as st
import numpy as np
from PIL import Image
import cv2 


# Load the cascade classifiers
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'navigation/models/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'navigation/models/haarcascade_eye.xml')
glasses_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'navigation/models/haarcascade_eye_tree_eyeglasses.xml')

st.title("Face Detection App")

st.markdown("""
Welcome to the Face Detection App! This app uses OpenCV to detect faces, eyes, and glasses in your photos.
Just take a picture using your webcam, and we'll do the rest!
""")

# Sidebar with options
st.sidebar.header("Detection Settings")
detect_faces = st.sidebar.checkbox("Detect Faces", value=True)
detect_eyes = st.sidebar.checkbox("Detect Eyes", value=True)
detect_glasses = st.sidebar.checkbox("Detect Glasses", value=True)

# Capture image using Streamlit's camera_input
st.subheader("Take a Picture")
picture = st.camera_input("")

if picture:
    st.image(picture, caption="Original Image", use_column_width=True)
    
    # Convert the picture to a format OpenCV can process
    image = Image.open(picture)
    image_np = np.array(image)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Detect faces
    if detect_faces:
        faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in faces:
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 4)
            cv2.putText(image_np, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Detect eyes
    if detect_eyes:
        eyes = eye_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in eyes:
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(image_np, 'Eye', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Detect glasses
    if detect_glasses:
        glasses = glasses_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
        for (x, y, w, h) in glasses:
            cv2.rectangle(image_np, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(image_np, 'Glasses', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Convert image_np back to an Image to display in Streamlit
    result_image = Image.fromarray(image_np)
    st.subheader("Processed Image")
    st.image(result_image, caption="Processed Image with Face, Eye, and Glasses Detection", use_column_width=True)

   
else:
    st.warning("Please take a picture to proceed with detection.")
