
# Military Object Detection App

## Overview
This repository hosts a Streamlit-based web application designed to detect military objects such as tanks, aircraft, and soldiers using a custom-trained YOLOv5 model. Beyond detection, it offers image analysis features including size, resolution, aspect ratio, and quality assessment, all wrapped in a sleek, user-friendly interface with real-time processing capabilities, ideal for military image analysis.

## Features
- **Object Detection**: Labels military objects with confidence scores using a pre-trained YOLOv5 model.
- **Image Analysis**: Provides metrics like height, width, aspect ratio, and quality status.
- **Interactive UI**: Features a modern design with custom CSS, built using Streamlit.
- **Real-Time Results**: Instant detection and analysis upon image upload.

## Prerequisites
- Python 3.8 or higher
- Git for repository cloning
- Required Python packages: `streamlit`, `opencv-python`, `numpy`, `torch`, `pillow`

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/military-object-detection.git
cd military-object-detection
```

### 2. Install Dependencies  
Set up a virtual environment (recommended) and install packages:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:
```bash
pip install streamlit opencv-python numpy torch pillow
```

### 3. Prepare the Model  
Place the trained YOLOv5 model file (`military_model_best.pt`) in the project directory (e.g., `D:/Final_project/military_model_best.pt`).  
Adjust the `model_path` in `app.py` if the location differs.

### 4. Run the App  
Start the app with:
```bash
streamlit run app.py
```
Access it in your browser at the local URL (e.g., `http://localhost:8501`).

## Usage
- Upload an image (.jpg, .png, .jpeg) via the sidebar.
- Click **"Detect Objects"** to view detection results.
- Check the right column for image analysis details.
- If errors arise, verify the file format and model path.

## File Structure
```
military-object-detection/
│
├── app.py                 # Main Streamlit application
├── military_model_best.pt  # Trained YOLOv5 model
├── README.md               # This file
└── requirements.txt        # (Optional) Dependency list
```
