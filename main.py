import cv2
import os
from plate_detector import detect_plate
from ocr import extract_text
import pandas as pd

# Load RTO mapping from CSV
rto_data = pd.read_csv("Indian RTO.csv", delimiter="\t", dtype=str)
rto_mapping = {row["RTO"]: (row["Location"], row["state"]) for _, row in rto_data.iterrows()}

def get_rto_details(plate_text):
    """Extract state and district from number plate."""
    rto_code = plate_text[:4]  # First 4 characters (e.g., 'CG04')
    return rto_mapping.get(rto_code, ("Unknown District", "Unknown State"))

def process_video(video_path, output_dir):
    """Detect number plates in a video and store unique ones."""
    cap = cv2.VideoCapture(video_path)
    detected_plates = set()
    frame_count = 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        plate_img = detect_plate(frame)
        if plate_img is not None:
            plate_text = extract_text(plate_img)
            if plate_text and plate_text not in detected_plates:
                detected_plates.add(plate_text)
                district, state = get_rto_details(plate_text)
                
                # Save image and output details
                filename = os.path.join(output_dir, f"{plate_text}.jpg")
                cv2.imwrite(filename, plate_img)
                print(f"Detected: {plate_text} | District: {district} | State: {state}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = "path_to_video.mp4"
output_dir = "detected_plates"
process_video(video_path, output_dir)
