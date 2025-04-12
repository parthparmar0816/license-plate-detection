# import cv2
# import torch
# from ultralytics import YOLO

# # Load YOLOv8 model
# model = YOLO("saved_models/indian_license_plate_detector2/weights/best.pt")  # Ensure you have a pre-trained YOLOv8 model

# def detect_license_plate(image_path):
#     image = cv2.imread(image_path)
    
#     # Run YOLO detection
#     results = model(image)
    
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
#             cropped_plate = image[y1:y2, x1:x2]  # Crop the plate region
            
#             # Save cropped license plate
#             cropped_path = "outputs/cropped_plate.jpg"
#             cv2.imwrite(cropped_path, cropped_plate)
            
#             return cropped_path  # Return cropped image path

#     return None


import cv2
import torch
from ultralytics import YOLO

# Load YOLOv8 model (Ensure you have a trained model in the correct path)
model = YOLO("saved_models/indian_license_plate_detector2/weights/best.pt")

def detect_license_plate(frame):
    """Detect license plate in a video frame using YOLOv8."""
    results = model.track(frame, persist=True, verbose=False)

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            cropped_plate = frame[y1:y2, x1:x2]  # Crop the plate region

            return cropped_plate  # Return cropped plate image

    return None  # No plate detected
