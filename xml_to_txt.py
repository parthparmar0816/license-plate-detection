import xml.etree.ElementTree as ET
import os

# Define the class mapping
classes = {'number_plate': 0}  # Assign class ID

def convert_bbox(size, box):
    """Convert (xmin, xmax, ymin, ymax) to YOLO format."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h

def convert_annotation(xml_file, output_dir):
    """Convert a single XML file to YOLO format."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    size = root.find("size")
    if size is None:
        return
    
    w = int(size.find("width").text)
    h = int(size.find("height").text)
    
    filename = os.path.basename(xml_file).replace(".xml", ".txt")
    txt_file = os.path.join(output_dir, filename)

    with open(txt_file, "w") as out_file:
        for obj in root.iter("object"):
            cls = obj.find("name").text
            if cls not in classes:
                continue  # Skip unknown classes

            cls_id = classes[cls]
            xmlbox = obj.find("bndbox")
            b = (
                float(xmlbox.find("xmin").text),
                float(xmlbox.find("xmax").text),
                float(xmlbox.find("ymin").text),
                float(xmlbox.find("ymax").text),
            )
            bb = convert_bbox((w, h), b)
            out_file.write(f"{cls_id} " + " ".join(map(str, bb)) + "\n")

def process_directory(xml_dir, output_dir):
    """Convert all XML files in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    for xml_file in os.listdir(xml_dir):
        if xml_file.endswith(".xml"):
            convert_annotation(os.path.join(xml_dir, xml_file), output_dir)

# Define paths
xml_annotations_path = "/home/jarvis/Desktop/ML/Number-Plate-Detection/training_data/number_plate_annos_ocr/number_plate_annos_ocr"
yolo_output_path = "/home/jarvis/Desktop/ML/Number-Plate-Detection/training_data/Annotations/labels"

# Run conversion
process_directory(xml_annotations_path, yolo_output_path)
