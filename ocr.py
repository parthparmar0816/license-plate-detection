import cv2
import numpy as np
from paddleocr import PaddleOCR
import re

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', rec_algorithm="SVTR_LCNet",drop_score=0.75,show_log=False)

def preprocess_image(image):
    """Enhance image for better OCR accuracy."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 75, 75)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    return binary

def is_valid_plate(text):
    """Check if text matches Indian license plate format."""
    pattern = r'^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{1,4}$'
    return bool(re.match(pattern, text))

def extract_text(image):
    """Extract best matching license plate from image."""
    if image is None or not isinstance(image, np.ndarray):
        return None, 0.0

    processed_image = preprocess_image(image)
    result = ocr.ocr(processed_image, cls=True)

    if not result or not result[0]:
        return None, 0.0

    text_candidates = []
    widths = []

    for line in result[0]:
        if line:
            bbox, (text, confidence) = line[0], line[1]
            width = abs(bbox[2][0] - bbox[0][0])
            clean_text = re.sub(r'[^A-Z0-9]', '', text.upper())
            if len(clean_text) >= 6:  # Basic filter to skip small junk
                text_candidates.append((clean_text, width, confidence))
                widths.append(width)

    if not text_candidates:
        return None, 0.0

    # Calculate width range thresholds (e.g., keep top 50% widest)
    widths = sorted(widths)
    median_width = widths[len(widths) // 2]
    min_width = 0.75 * median_width  # tweakable
    max_width = 2.0 * median_width  # tweakable

    filtered = [
        (text, width, conf) for (text, width, conf) in text_candidates
        if min_width <= width <= max_width
    ]

    # Further prioritize ones that match valid plate format
    plate_candidates = [(text, width, conf) for (text, width, conf) in filtered if is_valid_plate(text)]

    if plate_candidates:
        plate_candidates.sort(key=lambda x: x[1] * x[2], reverse=True)
        best_text, _, best_conf = plate_candidates[0]
    else:
        # fallback to largest good-looking candidate if none match pattern
        filtered.sort(key=lambda x: x[1] * x[2], reverse=True)
        best_text, _, best_conf = filtered[0]

    return best_text, best_conf
