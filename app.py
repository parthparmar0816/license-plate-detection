import streamlit as st
import cv2
import tempfile
import os
import numpy as np
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

# Upload section
upload_option = st.radio("Choose media type:", ["Image", "Video"])

if upload_option == "Image":
    uploaded_images = st.file_uploader("Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_images:
        for uploaded_file in uploaded_images:
            file_bytes = uploaded_file.read()
            np_arr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            print("Detecting from Image")
            plate_img = detect_license_plate(image)
            print("Done")
            if plate_img is not None:
                plate_text, confidence = extract_text(plate_img) or (None, 0.0)
                if plate_text:
                    plate_text = plate_text.replace(" ", "").upper()
                    district, state = get_rto_details(plate_text)
                    filename = f"{OUTPUT_DIR}/{plate_text}.png"
                    cv2.imwrite(filename, plate_img)

                    with st.container():
                        st.image(plate_img, caption=f"**{plate_text}** → {district}, {state} (Conf: {confidence:.2f})", use_container_width=True)
            else:
                st.warning(f"No plate detected in {uploaded_file.name}")

elif upload_option == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(uploaded_file.read())
            temp_video_path = temp_video.name  

        st.write(f"📹 Processing video: {uploaded_file.name}")  

        detected_plates = {}  # Store best unique plates {plate_text: (plate_img, confidence)}
        cap = cv2.VideoCapture(temp_video_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_interval = max(fps, 1)  # Process 1 frame every second

        frame_number = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % frame_interval == 0:
                plate_img = detect_license_plate(frame)
                if plate_img is not None:
                    plate_text, confidence = extract_text(plate_img) or (None, 0.0)

                    if plate_text:
                        plate_text = plate_text.replace(" ", "").upper()
                        if plate_text not in detected_plates or confidence > detected_plates[plate_text][1]:
                            detected_plates[plate_text] = (plate_img, confidence)

            frame_number += 1

        cap.release()
        os.remove(temp_video_path)

        for plate_text, (plate_img, confidence) in detected_plates.items():
            district, state = get_rto_details(plate_text)
            filename = f"{OUTPUT_DIR}/{plate_text}.png"
            cv2.imwrite(filename, plate_img)

            with st.container():
                st.image(plate_img, caption=f"**{plate_text}** → {district}, {state} (Conf: {confidence:.2f})", use_container_width=True)
