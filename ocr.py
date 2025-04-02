import cv2
import numpy as np
from paddleocr import PaddleOCR

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', rec_algorithm="CRNN")

def preprocess_image(image):
    """Enhance image for better OCR accuracy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray = cv2.bilateralFilter(gray, 13, 75, 75)  # Reduce noise while keeping edges sharp

    # Apply contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Apply adaptive thresholding for binarization
    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    return binary

def extract_text(image):
    """Extract only the largest detected text using PaddleOCR."""
    if image is None or not isinstance(image, np.ndarray):
        return None, 0.0  # Return default confidence as 0.0

    processed_image = preprocess_image(image)
    result = ocr.ocr(processed_image, cls=True)

    if not result or result[0] is None:
        return None, 0.0  # Return default confidence as 0.0

    # Extract text along with bounding box width
    text_candidates = []
    for line in result[0]:
        if line:
            bbox, (text, confidence) = line[0], line[1]
            width = abs(bbox[2][0] - bbox[0][0])  # Calculate bounding box width
            text_candidates.append((text, width, confidence))

    # Sort by width (largest text first)
    text_candidates.sort(key=lambda x: x[1], reverse=True)

    return (text_candidates[0][0], text_candidates[0][2]) if text_candidates else (None, 0.0)
