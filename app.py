import streamlit as st
import cv2
import tempfile
import os
import pandas as pd
from plate_detector import detect_license_plate
from ocr import extract_text
from state_codes import get_rto_details

# Set up output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load RTO mapping from CSV
@st.cache_data
def load_rto_data():
    rto_data = pd.read_csv("Indian_RTO.csv", delimiter=",", dtype=str)
    rto_data.columns = rto_data.columns.str.replace('"', '').str.strip()
    return {row["RTO"]: (row["Location"], row["state"]) for _, row in rto_data.iterrows()}

rto_mapping = load_rto_data()

# Streamlit UI
st.title("🔍 Number Plate Detection & Recognition")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(uploaded_file.read())
        temp_video_path = temp_video.name  

    st.write(f"📹 Processing video: {uploaded_file.name}")  

    detected_plates = {}  # Store best unique plates {plate_text: (plate_img, confidence)}
    cap = cv2.VideoCapture(temp_video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get video FPS
    frame_interval = max(fps, 1)  # Process 1 frame per second (avoid division by zero)

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        frame_count += 1
        if frame_count % frame_interval != 0:  # Process only 1 frame per second
            continue

        plate_img = detect_license_plate(frame)
        if plate_img is not None:
            plate_text, confidence = extract_text(plate_img) or ("UNKNOWN", 0.0)  # Ensure valid values
            plate_text = plate_text.replace(" ", "").upper()  # Standardize format

            # If it's a new plate OR has better confidence, store it
            if plate_text and (plate_text not in detected_plates or confidence > detected_plates[plate_text][1]):
                detected_plates[plate_text] = (plate_img, confidence)  # Keep best version

    cap.release()
    os.remove(temp_video_path)

    # 🔹 Display best unique detections!
    for plate_text, (plate_img, confidence) in detected_plates.items():
        district, state = get_rto_details(plate_text)
        filename = f"{OUTPUT_DIR}/{plate_text}.png"
        cv2.imwrite(filename, plate_img)

        with st.container():
            st.image(plate_img, caption=f"**{plate_text}** → {district}, {state} (Conf: {confidence:.2f})", use_container_width=True)
