"""Convert annotation data from XML format to TXT format.
"""

import os
import xml.etree.ElementTree as ET


xml_dir = r"%HOMEDRIVE%%HOMEPATH%\Desktop\xml2txt\Annotations_pascal_xml"  # Directory with XML files
txt_dir = r"%HOMEDRIVE%%HOMEPATH%\Desktop\xml2txt\Annotations_yolo_txt"    # Directory for outputting YOLO format text files

if not os.path.exists(txt_dir):
    os.makedirs(txt_dir)

# List of classes
classes = []

# Scan files within xml_dir
for filename in os.listdir(xml_dir):
    # Only files with the .xml extension will be processed.
    if filename.endswith(".xml"):
        # XML file path
        xml_path = os.path.join(xml_dir, filename)
        
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image size
        size = root.find("size")
        if size is None:
            # Skip if there is no size tag
            continue
        
        width = float(size.find("width").text)
        height = float(size.find("height").text)
        
        # Path to the output text file (replace with .txt)
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(txt_dir, txt_filename)
        
        # List of YOLO-style annotations
        yolo_annotations = []
        
        # Get all object tags and convert them to YOLO format
        for obj in root.findall("object"):
            # Class name
            class_name = obj.find("name").text
            if class_name not in classes:
                classes.append(class_name)
            class_id = classes.index(class_name)
            
            # Bounding box coordinates
            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            
            # The YOLO format is (class_id, x_center, y_center, w, h) [normalized]
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            
            # Adjust to format the number of decimal places before outputting.
            annotation_str = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            yolo_annotations.append(annotation_str)
        
        # Write to a text file
        with open(txt_path, "w", encoding="utf-8") as f:
            for line in yolo_annotations:
                f.write(line + "\n")

# After the conversion is complete, output a list of classes stored in the `classes` directory
print("Class List:", classes)
