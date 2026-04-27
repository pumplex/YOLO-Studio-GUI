"""Precision、Recall、F-measure/F1-score
"""

import os
from pathlib import Path
import shutil
import csv
from ultralytics import YOLO
import numpy as np
import cv2
import random
from typing import Dict, List, Tuple

# Settings/Config
DATASET_DIR = r"C:\Users\he81t\ubuntu\images\green_soybeans\shonai1_test_data"  # Dataset directory
MODEL_PATH = r"C:\Users\he81t\ubuntu\images\green_soybeans\shonai3_models\weights\best.pt"  # YOLO model path
CONF_THRESHOLD = 0.5  # confidence threshold

def create_output_dirs():
    """Create Output Folder"""
    base_dir = Path(DATASET_DIR) / "test_results"
    detect_dir = base_dir / "detection_images"
    
    base_dir.mkdir(exist_ok=True)
    detect_dir.mkdir(exist_ok=True)
    
    return base_dir, detect_dir

def generate_colors(num_classes: int) -> Dict[int, Tuple[int, int, int]]:
    """Generate a fixed RGB color for each classification class"""
    random.seed(42)
    colors = {}
    for i in range(num_classes):
        colors[i] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    return colors

def calculate_iou(box1, box2):
    """Calculate the IoU between two bounding boxes"""
    # Convert to (x1, y1, x2, y2) format
    b1_x1, b1_y1 = box1[0] - box1[2]/2, box1[1] - box1[3]/2
    b1_x2, b1_y2 = box1[0] + box1[2]/2, box1[1] + box1[3]/2
    b2_x1, b2_y1 = box2[0] - box2[2]/2, box2[1] - box2[3]/2
    b2_x2, b2_y2 = box2[0] + box2[2]/2, box2[1] + box2[3]/2
    
    # Intersection area
    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)
    
    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    
    # Union area
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    
    return inter_area / (b1_area + b2_area - inter_area)

def evaluate_detection(gt_boxes, gt_classes, pred_boxes, pred_classes, iou_threshold=0.5):
    """The detection results are evaluated, and Precision, Recall, and F-score are calculated."""
    if len(pred_boxes) == 0:
        if len(gt_boxes) == 0:
            return 1.0, 1.0, 1.0
        return 0.0, 0.0, 0.0
    
    if len(gt_boxes) == 0:
        return 0.0, 0.0, 0.0
    
    true_positives = 0
    used_gt = set()
    
    for pred_idx, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        
        for i, gt in enumerate(gt_boxes):
            if i in used_gt:
                continue
                
            iou = calculate_iou(pred, gt)
            # Only cases where the class matches and the IoU is above the threshold will be considered
            if iou > best_iou and pred_classes[pred_idx] == gt_classes[i]:
                best_iou = iou
                best_gt_idx = i
        
        if best_iou >= iou_threshold:
            true_positives += 1
            used_gt.add(best_gt_idx)
    
    precision = true_positives / len(pred_boxes)
    recall = true_positives / len(gt_boxes)
    
    if precision + recall == 0:
        f_value = 0.0
    else:
        f_value = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f_value

def main():
    model = YOLO(MODEL_PATH)
    num_classes = len(model.names)
    class_colors = generate_colors(num_classes)
    class_names = model.names
    
    base_dir, detect_dir = create_output_dirs()
    image_files = [f for f in os.listdir(DATASET_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Create a data frame for counting the number of detections
    detection_counts = []
    
    total_precision = 0
    total_recall = 0
    total_f_value = 0
    
    for img_file in image_files:
        img_path = os.path.join(DATASET_DIR, img_file)
        label_path = os.path.join(DATASET_DIR, os.path.splitext(img_file)[0] + '.txt')
        
        if not os.path.exists(label_path):
            continue
        
        gt_boxes = []
        gt_classes = []
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x, y, w, h = map(float, line.strip().split())
                gt_boxes.append([x, y, w, h])
                gt_classes.append(int(class_id))
        
        results = model(img_path, conf=CONF_THRESHOLD)[0]
        img = cv2.imread(img_path)
        pred_boxes = []
        pred_classes = []
        
        # Count the number of detections for each class
        class_counts = {i: 0 for i in range(num_classes)}
        
        for box in results.boxes:
            if float(box.conf[0]) < CONF_THRESHOLD:
                continue
                
            x, y, w, h = box.xywh[0].tolist()
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            color = class_colors[class_id]
            
            # Detection count
            class_counts[class_id] += 1
            
            norm_x = x / img.shape[1]
            norm_y = y / img.shape[0]
            norm_w = w / img.shape[1]
            norm_h = h / img.shape[0]
            pred_boxes.append([norm_x, norm_y, norm_w, norm_h])
            pred_classes.append(class_id)
            
            cv2.rectangle(img, 
                         (int(x-w/2), int(y-h/2)), 
                         (int(x+w/2), int(y+h/2)), 
                         color, 
                         2)
            
            label = f"{model.names[class_id]} {conf:.2f}"
            cv2.putText(img, 
                        label, 
                        (int(x-w/2), int(y-h/2)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2)
        
        # Add detection count to list
        detection_count = [img_file] + [class_counts[i] for i in range(num_classes)]
        detection_counts.append(detection_count)
        
        precision, recall, f_value = evaluate_detection(gt_boxes, gt_classes, pred_boxes, pred_classes)
        
        total_precision += precision
        total_recall += recall
        total_f_value += f_value
        
        output_img_path = str(detect_dir / f"{os.path.splitext(img_file)[0]}_det.jpg")
        cv2.imwrite(output_img_path, img)
        
        result_txt = detect_dir / f"{os.path.splitext(img_file)[0]}_det.txt"
        with open(result_txt, 'w') as f:
            for i, box in enumerate(pred_boxes):
                f.write(f"{pred_classes[i]} {' '.join(map(str, box))}\n")
    
    # Save the number of detections to a CSV file.
    detection_csv_path = base_dir / "num_of_detections.csv"
    with open(detection_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['filename'] + [class_names[i] for i in range(num_classes)]
        writer.writerow(header)
        writer.writerows(detection_counts)
    
    n_images = len(image_files)
    avg_precision = total_precision / n_images
    avg_recall = total_recall / n_images
    avg_f_value = total_f_value / n_images
    
    # csv_path = base_dir / "evaluation_results.csv"
    # with open(csv_path, 'w', newline='') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(['Metric', 'Value'])
    #     writer.writerow(['Average Precision', avg_precision])
    #     writer.writerow(['Average Recall', avg_recall])
    #     writer.writerow(['Average F-value', avg_f_value])
    
    txt_path = base_dir / "precision_recall_f-value.txt"
    with open(txt_path, 'w') as f:
        f.write(f"Average Precision: {avg_precision:.4f}\n")
        f.write(f"Average Recall: {avg_recall:.4f}\n")
        f.write(f"Average F-value: {avg_f_value:.4f}\n")

if __name__ == "__main__":
    main()
