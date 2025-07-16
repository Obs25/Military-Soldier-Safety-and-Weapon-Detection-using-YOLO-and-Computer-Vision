import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import os

st.set_page_config(page_title="Military Object Detector", layout="wide")

st.markdown(
    """
    <style>
    .main {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    .stButton>button {background-color: #2e7d32; color: white; border-radius: 5px; padding: 10px 20px; font-size: 16px;}
    .stButton>button:hover {background-color: #1b5e20;}
    .stHeader {color: #d32f2f; font-size: 36px; font-weight: bold; text-align: center; margin-bottom: 20px;}
    .stSubheader {color: #424242; font-size: 24px;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="stHeader">Military Object Detection</div>', unsafe_allow_html=True)

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

@st.cache_resource
def load_model():
    model_path = 'D:/Final_project/military_model_best.pt'
    if not os.path.exists(model_path):
        st.error("Model file not found! Please ensure the path is correct.")
        return None
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    return model

model = load_model()

if model is not None and uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.markdown('<div class="stSubheader">Detection Results</div>', unsafe_allow_html=True)
    if st.button("Detect Objects"):
        with st.spinner("Detecting..."):
            results = model(img_cv)
            detected_img = results.render()[0]
            st.image(detected_img, caption="Detected Objects", use_column_width=True)
            st.write("### Detection Details")
            for det in results.xyxy[0]:
                class_id = int(det[5])
                confidence = det[4]
                class_names = ['camouflage_soldier', 'weapon', 'military_tank', 'military_truck',
                             'military_vehicle', 'civilian', 'soldier', 'civilian_vehicle',
                             'military_artillery', 'military_aircraft', 'military_warship']
                st.write(f"- {class_names[class_id]} (Confidence: {confidence:.2f})")

elif uploaded_file is None:
    st.info("Please upload an image to start detection.")
elif model is None:
    st.error("Model loading failed. Check the console for errors.")

st.markdown(
    """
    <div style='text-align: center; color: #757575; margin-top: 20px;'>
        Powered by YOLOv5 & Streamlit | © 2025
    </div>
    """,
    unsafe_allow_html=True
)