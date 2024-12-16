import numpy as np
import matplotlib.pyplot as plt
import random
import os
import cv2
import shutil
import tqdm
import glob
import torch
from ultralytics import YOLO
import streamlit as st


# streamlit headers
st.title('Dynamic Avatar Clothing App')
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# importing model and defining prediction pipeline
# Load the pre-trained model
def load_model():
    detection_model = YOLO("best.pt")
    return detection_model

# Prediction pipeline
def predict_image(model, image_path):
    results = model.predict(source=image_path, conf=0.4, save=True, line_width=2)
    return results

# Function to get the dominant color within a bounding box
def get_dominant_color(image, box):
    x1, y1, x2, y2 = map(int, box)
    cropped_image = image[y1:y2, x1:x2]
    pixels = cropped_image.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 2  # Number of clusters (colors)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    dominant_color = centers[np.argmax(np.unique(labels, return_counts=True)[1])]
    return tuple(map(int, dominant_color))





if uploaded_file:

    model = load_model()
    
    # Convert the file to an OpenCV image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    
    st.image(opencv_image, channels="BGR", caption="Uploaded Image")

    if st.button('Predict'):
        results = predict_image(model, opencv_image)
        predictions = []
        for result in results:
            for box in result.boxes:
                box_coords = box.xyxy[0].tolist()
                class_name = model.names[int(box.cls[0])]
                dominant_color = get_dominant_color(opencv_image, box_coords)
                predictions.append({
                  'box_coordinates': box_coords,
                  'class_name': class_name,
                  'dominant_color': dominant_color
                })
        
        for pred in predictions:
            st.write(f"**Class Name: {pred['class_name']}**")
            st.write(f"Box Coordinates: {pred['box_coordinates']}")
            st.write(f"Dominant Color: {pred['dominant_color']}")

        st.image('runs/detect/predict/image0.jpg', channels="BGR", caption="Uploaded Image")
        # Delete the prediction directory
        shutil.rmtree('runs/detect/', ignore_errors=True)            