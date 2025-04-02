from ultralytics import YOLO

# Load a YOLOv8 model (Pretrained on COCO)
model = YOLO("yolov8s.pt")  # You can also use yolov8s.pt, yolov8m.pt, yolov8l.pt

# Train the model
model.train(
    data="/home/jarvis/Desktop/ML/Number-Plate-Detection/dataset/data.yaml",  # Path to dataset config
    epochs=30,  # Number of training epochs
    imgsz=512,  # Image size (adjust if needed)
    batch=4,  # Batch size (adjust based on GPU memory)
    workers=2,  # Number of worker threads
    lr0=0.0005,  # Learning rate
    project="saved_models",  # Custom directory for model saving
    name="indian_license_plate_detector",  # Custom model name
    save=True  # Ensure the model is saved
)


# import torch
# torch.cuda.empty_cache()
