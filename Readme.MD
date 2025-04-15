
# License Plate Detection (Indian Scenario)

This project focuses on detecting and recognizing Indian vehicle license plates using deep learning techniques.
It leverages the YOLOv5 architecture for object detection and EasyOCR for character recognition.

## Features

- **License Plate Detection**: Utilizes YOLOv5 for accurate detection of license plates in images and videos.
- **Optical Character Recognition (OCR)**: Employs EasyOCR to extract alphanumeric characters from detected license plates.
- **Indian RTO Mapping**: Maps detected license plate codes to their corresponding Indian Regional Transport Offices (RTOs) using `Indian_RTO.csv`.

## Project Structure

```
license-plate-detection/
├── saved_models/           # Directory containing trained model weights
├── app.py                  # Streamlit application for real-time detection
├── main.py                 # Entry point for the application
├── ocr.py                  # Module for OCR processing using EasyOCR
├── plate_detector.py       # Module for detecting license plates using YOLOv5
├── train.py                # Script for training the YOLOv5 model
├── xml_to_txt.py           # Utility to convert XML annotations to YOLO format
├── Indian_RTO.csv          # CSV file mapping license plate codes to RTOs
├── state_codes.py          # Python dictionary mapping state codes
├── requirements.txt        # List of required Python packages
└── .gitignore              # Git ignore file
```

## Installation

1. **Clone the Repository**

```bash
git clone https://github.com/parthparmar0816/license-plate-detection.git
cd license-plate-detection
```

2. **Set Up a Virtual Environment (Optional but Recommended)(Ubuntu)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

To train the YOLOv8 model on your dataset:

```bash
python train.py
```

Ensure your dataset is organized in the YOLO format and the paths are correctly set in the script.

### Running the Application

To start the Streamlit application for real-time license plate detection:

```bash
streamlit run app.py
```

This will launch a web interface where you can upload images or use your webcam for detection.

## Dataset

The model is trained on a custom dataset tailored for Indian license plates.
Annotations are converted to YOLO format using `xml_to_txt.py`.

## Acknowledgements

- [YOLOv8 by Ultralytics](https://docs.ultralytics.com/models/yolov8/)
- [Dataset used to Finetune Yolo Model](https://www.kaggle.com/datasets/dataclusterlabs/indian-number-plates-dataset?resource=download)
## License

This project is licensed under the MIT License.