
import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image
import os

# Set page config for attractive layout
st.set_page_config(page_title="Military Object Detection", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for enhanced styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f4f8;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #1976d2;
        color: white;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
    }
    .stHeader {
        color: #d32f2f;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 25px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
    }
    .stSubheader {
        color: #424242;
        font-size: 24px;
        margin-bottom: 15px;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 5px;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title
st.markdown('<div class="stHeader">Military Object Detection</div>', unsafe_allow_html=True)

# Sidebar for mode selection and file upload
st.sidebar.header("Options")
mode = st.sidebar.selectbox("Select Mode", ["Object Detection", "Image Processing"])
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], key="file_uploader")

# Initialize YOLOv5 model
@st.cache_resource
def load_model():
    model_path = 'D:/Final_project/military_model_best.pt'
    if not os.path.exists(model_path):
        st.error("Model file not found! Please ensure the path is correct.")
        return None
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
    return model

model = load_model()

# Function to process and transform the sample image
def process_sample_image(image_path):
    if os.path.exists(image_path):
        # Read the original image
        img = cv2.imread(image_path)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Horizontal flip
        flipped_img = cv2.flip(img, 1)
        flipped_pil = Image.fromarray(cv2.cvtColor(flipped_img, cv2.COLOR_BGR2RGB))
        
        # Rotate 90 degrees clockwise
        rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        rotated_pil = Image.fromarray(cv2.cvtColor(rotated_img, cv2.COLOR_BGR2RGB))
        
        return img_pil, flipped_pil, rotated_pil
    else:
        st.error("Sample image not found! Check the path.")
        return None, None, None

# Main content with dynamic display based on mode
if mode == "Object Detection":
    if model is not None and uploaded_file is not None:
        st.markdown('<div class="stSubheader">Uploaded Image & Detection</div>', unsafe_allow_html=True)
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Detect objects
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if st.button("Detect Objects"):
            with st.spinner("Detecting..."):
                results = model(img_cv)
                detected_img = results.render()[0]
                st.image(detected_img, caption="Detected Objects", use_container_width=True)
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

elif mode == "Image Processing":
    st.markdown('<div class="stSubheader">Training Dataset Image Processing</div>', unsafe_allow_html=True)
    sample_path = "D:/Final_project/military_object_dataset/subset_train/images/000083.jpg"
    original_img, flipped_img, rotated_img = process_sample_image(sample_path)
    
    if original_img is not None:
        st.write("### Original Image (Sample)")
        st.image(original_img, caption="Original Image", use_container_width=True)

        st.write("### Flipped Image (Horizontal)")
        st.image(flipped_img, caption="Horizontally Flipped", use_container_width=True)

        st.write("### Rotated Image (90°)")
        st.image(rotated_img, caption="Rotated 90°", use_container_width=True)

# Footer
st.markdown(
    """
    <div style='text-align: center; color: #757575; margin-top: 30px; font-size: 14px;'>
        Powered by YOLOv5 & Streamlit | © 2025
    </div>
    """,
    unsafe_allow_html=True
)
