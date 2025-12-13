#!/usr/bin/env python3
"""
EcoWheels Dataset Balancer
Fixes class imbalance through smart augmentation
"""

import os
import shutil
import random
import yaml
from pathlib import Path
from collections import Counter
from datetime import datetime
import cv2
import albumentations as A
import numpy as np

# ================= CONFIGURATION =================
INPUT_DIR = "/home/immaculatapatrickumoh/Documents/EcoWheels_Proj/data"
OUTPUT_DIR = "/home/immaculatapatrickumoh/Documents/EcoWheels_Proj/balanced_data"
TARGET_SAMPLES_PER_CLASS = 500
CLASS_NAMES = ["Cardboard", "Glass", "Metal", "Mixed Waste", "Organic Waste", "Paper", "Plastic", "Textiles"]
# ==================================================

def read_yolo_label(label_path, img_width, img_height):
    """Read YOLO format label file"""
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    annotations = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            class_id = int(float(parts[0]))
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Ensure coordinates are valid
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.001, min(1.0, width))
            height = max(0.001, min(1.0, height))
            
            # Convert to pixel coordinates
            x_min = max(0.0, (x_center - width/2)) * img_width
            y_min = max(0.0, (y_center - height/2)) * img_height
            x_max = min(1.0, (x_center + width/2)) * img_width
            y_max = min(1.0, (y_center + height/2)) * img_height
            
            if x_max > x_min and y_max > y_min:
                annotations.append({
                    'class_id': class_id,
                    'bbox': [x_min, y_min, x_max, y_max]
                })
    
    return annotations

def write_yolo_label(label_path, annotations, img_width, img_height):
    """Write YOLO format label file"""
    with open(label_path, 'w') as f:
        for ann in annotations:
            x_min, y_min, x_max, y_max = ann['bbox']
            
            x_center = ((x_min + x_max) / 2) / img_width
            y_center = ((y_min + y_max) / 2) / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            
            # Ensure valid ranges
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.001, min(1.0, width))
            height = max(0.001, min(1.0, height))
            
            f.write(f"{ann['class_id']} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def get_augmentation_pipeline(class_id, current_count, target_count):
    """Get augmentation pipeline based on class imbalance"""
    augmentation_factor = target_count / max(current_count, 1)
    
    # Heavy augmentation for minority classes
    if augmentation_factor > 5:
        return A.Compose([
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.7),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=45, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.4, p=0.7),
            A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=40, val_shift_limit=30, p=0.7),
            A.Blur(blur_limit=5, p=0.3),
            A.CLAHE(clip_limit=4.0, p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    # Moderate augmentation
    elif augmentation_factor > 2:
        return A.Compose([
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))
    
    # Light augmentation
    else:
        return A.Compose([
            A.Resize(640, 640),
            A.HorizontalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def split_validation_set(output_dir, split_ratio=0.2):
    """Split training data for validation if needed"""
    train_img_dir = Path(output_dir) / "train" / "images"
    train_lbl_dir = Path(output_dir) / "train" / "labels"
    val_img_dir = Path(output_dir) / "val" / "images"
    val_lbl_dir = Path(output_dir) / "val" / "labels"
    
    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    train_images = list(train_img_dir.glob("*"))
    if len(train_images) == 0:
        return 0
    
    num_val = max(1, int(len(train_images) * split_ratio))
    val_images = random.sample(train_images, num_val)
    
    moved = 0
    for img_file in val_images:
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Move image
            dest_img = val_img_dir / img_file.name
            shutil.move(str(img_file), str(dest_img))
            
            # Move label
            label_file = train_lbl_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                dest_lbl = val_lbl_dir / f"{img_file.stem}.txt"
                shutil.move(str(label_file), str(dest_lbl))
                moved += 1
    
    return moved

def balance_dataset():
    """Main function to balance the dataset"""
    
    # Create output directories
    output_train_img = Path(OUTPUT_DIR) / "train" / "images"
    output_train_lbl = Path(OUTPUT_DIR) / "train" / "labels"
    output_val_img = Path(OUTPUT_DIR) / "val" / "images"
    output_val_lbl = Path(OUTPUT_DIR) / "val" / "labels"
    
    for dir_path in [output_train_img, output_train_lbl, output_val_img, output_val_lbl]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Analyze current distribution
    print("Analyzing dataset distribution...")
    
    # Use YOUR actual paths
    input_train_lbl = Path(INPUT_DIR) / "train" / "labels"
    input_val_lbl = Path(INPUT_DIR) / "validation" / "labels"  # Changed from "val" to "validation"
    input_train_img = Path(INPUT_DIR) / "train" / "images"
    input_val_img = Path(INPUT_DIR) / "validation" / "images"   # Changed from "val" to "validation"
    
    # Count current class distribution
    class_counts = Counter()
    image_class_map = {}
    
    print(f"Looking for training labels in: {input_train_lbl}")
    print(f"Looking for validation labels in: {input_val_lbl}")
    
    # Process training labels
    for label_file in input_train_lbl.glob("*.txt"):
        try:
            with open(label_file, 'r') as f:
                classes_in_image = []
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(float(line.split()[0]))
                        class_counts[class_id] += 1
                        classes_in_image.append(class_id)
                
                if classes_in_image:
                    image_class_map[label_file.stem] = classes_in_image
        except Exception as e:
            print(f"Warning: Could not read {label_file}: {e}")
    
    print(f"\nCurrent class distribution:")
    for class_id in sorted(class_counts.keys()):
        if class_id < len(CLASS_NAMES):
            print(f"  Class {class_id} ({CLASS_NAMES[class_id]}): {class_counts[class_id]} samples")
        else:
            print(f"  Class {class_id} (Unknown): {class_counts[class_id]} samples")
    
    print(f"\nTotal training images: {len(image_class_map)}")
    print(f"Total annotations: {sum(class_counts.values())}")
    
    # Calculate augmentation factors
    augmentation_factors = {}
    for class_id in range(len(CLASS_NAMES)):
        current = class_counts.get(class_id, 0)
        if current == 0:
            augmentation_factors[class_id] = 0
        else:
            augmentation_factors[class_id] = TARGET_SAMPLES_PER_CLASS / current
    
    print("\nAugmentation factors needed:")
    for class_id, factor in sorted(augmentation_factors.items()):
        if factor > 0:
            print(f"  Class {class_id}: {factor:.1f}x (from {class_counts.get(class_id, 0)} to ~{TARGET_SAMPLES_PER_CLASS})")
    
    # ================== PROCESS TRAINING SET ==================
    print("\n" + "="*50)
    print("Processing training set...")
    
    # Copy all original training images
    copied_count = 0
    for img_file in input_train_img.glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            label_file = input_train_lbl / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy(img_file, output_train_img / img_file.name)
                shutil.copy(label_file, output_train_lbl / label_file.name)
                copied_count += 1
    
    print(f"Copied {copied_count} original training images")
    
    # Create augmented versions
    augmented_count = 0
    for class_id in range(len(CLASS_NAMES)):
        current_count = class_counts.get(class_id, 0)
        if current_count == 0:
            print(f"Skipping Class {class_id} - no samples")
            continue
        
        factor = augmentation_factors[class_id]
        if factor <= 1:
            print(f"Skipping Class {class_id} - already has enough samples ({current_count})")
            continue
        
        # Find images containing this class
        images_with_class = []
        for img_name, classes in image_class_map.items():
            if class_id in classes:
                images_with_class.append(img_name)
        
        if not images_with_class:
            print(f"Warning: No images found for Class {class_id}")
            continue
        
        # Create augmented versions
        needed_augmentations = int(TARGET_SAMPLES_PER_CLASS - current_count)
        print(f"\nClass {class_id}: Creating {needed_augmentations} augmented samples...")
        
        augment = get_augmentation_pipeline(class_id, current_count, TARGET_SAMPLES_PER_CLASS)
        
        created = 0
        attempts = 0
        max_attempts = needed_augmentations * 3
        
        while created < needed_augmentations and attempts < max_attempts:
            attempts += 1
            img_name = random.choice(images_with_class)
            
            # Find image file
            img_file = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                test_file = input_train_img / f"{img_name}{ext}"
                if test_file.exists():
                    img_file = test_file
                    break
            
            if not img_file:
                continue
            
            label_file = input_train_lbl / f"{img_name}.txt"
            if not label_file.exists():
                continue
            
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            img_height, img_width = img.shape[:2]
            
            # Read annotations
            annotations = read_yolo_label(label_file, img_width, img_height)
            
            # Filter for this class
            class_annotations = [ann for ann in annotations if ann['class_id'] == class_id]
            other_annotations = [ann for ann in annotations if ann['class_id'] != class_id]
            
            if not class_annotations:
                continue
            
            # Prepare for augmentation
            bboxes = []
            class_labels = []
            
            for ann in class_annotations:
                bboxes.append(ann['bbox'])
                class_labels.append(ann['class_id'])
            
            # Apply augmentation
            try:
                augmented = augment(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_labels
                )
                
                # Save augmented image
                aug_img_name = f"{img_name}_aug{class_id}_{created:04d}.jpg"
                cv2.imwrite(str(output_train_img / aug_img_name), augmented['image'])
                
                # Combine annotations
                all_annotations = []
                
                # Add augmented bboxes
                for bbox, label in zip(augmented['bboxes'], augmented['class_labels']):
                    all_annotations.append({
                        'class_id': label,
                        'bbox': bbox
                    })
                
                # Add original other annotations
                for ann in other_annotations:
                    all_annotations.append(ann)
                
                # Write label
                write_yolo_label(
                    output_train_lbl / f"{aug_img_name[:-4]}.txt",
                    all_annotations,
                    augmented['image'].shape[1],
                    augmented['image'].shape[0]
                )
                
                created += 1
                augmented_count += 1
                
                if created % 50 == 0:
                    print(f"  Created {created}/{needed_augmentations} augmentations...")
                    
            except Exception as e:
                continue
    
    # ================== PROCESS VALIDATION SET ==================
    print("\n" + "="*50)
    print("Processing validation set...")
    
    val_copied = 0
    if input_val_img.exists() and input_val_lbl.exists():
        for img_file in input_val_img.glob("*"):
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                label_file = input_val_lbl / f"{img_file.stem}.txt"
                if label_file.exists():
                    shutil.copy(img_file, output_val_img / img_file.name)
                    shutil.copy(label_file, output_val_lbl / label_file.name)
                    val_copied += 1
        print(f"Copied {val_copied} validation images")
    else:
        print(f"Validation folder not found at: {input_val_img}")
        print("Will split training data for validation...")
        val_copied = split_validation_set(OUTPUT_DIR, 0.2)
        print(f"Created {val_copied} validation images from training split")
    
    # ================== CREATE DATASET.YAML ==================
    print("\n" + "="*50)
    print("Creating dataset.yaml...")
    
    dataset_yaml = {
        'path': str(Path(OUTPUT_DIR).absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(CLASS_NAMES)},
        'nc': len(CLASS_NAMES)
    }
    
    yaml_path = Path(OUTPUT_DIR) / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created {yaml_path}")
    
    # ================== FINAL STATISTICS ==================
    print("\n" + "="*50)
    print("FINAL STATISTICS:")
    print("="*50)
    
    # Count new distribution
    new_class_counts = Counter()
    for label_file in output_train_lbl.glob("*.txt"):
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(float(line.split()[0]))
                        new_class_counts[class_id] += 1
        except:
            continue
    
    print("\nNEW class distribution (training):")
    total_new = 0
    for class_id in sorted(new_class_counts.keys()):
        count = new_class_counts[class_id]
        total_new += count
        if class_id < len(CLASS_NAMES):
            print(f"  Class {class_id} ({CLASS_NAMES[class_id]}): {count} samples")
        else:
            print(f"  Class {class_id} (Unknown): {count} samples")
    
    train_images = len(list(output_train_img.glob("*")))
    val_images = len(list(output_val_img.glob("*")))
    
    print(f"\nTotal training images: {train_images}")
    print(f"Total validation images: {val_images}")
    print(f"Total annotations: {total_new}")
    print(f"Original annotations: {sum(class_counts.values())}")
    print(f"Augmentations created: {augmented_count}")
    
    # Save report
    report_path = Path(OUTPUT_DIR) / "balancing_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"EcoWheels Balancing Report - {datetime.now()}\n")
        f.write("="*50 + "\n\n")
        f.write("Original Distribution:\n")
        for class_id in sorted(class_counts.keys()):
            f.write(f"  Class {class_id}: {class_counts[class_id]}\n")
        f.write(f"\nTotal: {sum(class_counts.values())}\n\n")
        
        f.write("New Distribution:\n")
        for class_id in sorted(new_class_counts.keys()):
            f.write(f"  Class {class_id}: {new_class_counts[class_id]}\n")
        f.write(f"\nTotal: {total_new}\n")
        f.write(f"Augmentations: {augmented_count}\n")
    
    print(f"\n📄 Report saved: {report_path}")
    print(f"\n✅ Dataset balancing completed!")
    print(f"📁 Output: {OUTPUT_DIR}")
    
    return yaml_path

if __name__ == "__main__":
    print("EcoWheels Dataset Balancer")
    print("="*40)
    
    # Check dependencies
    try:
        import albumentations
        import yaml
        import cv2
    except ImportError:
        print("Installing dependencies...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'albumentations', 'pyyaml', 'opencv-python'])
    
    yaml_path = balance_dataset()
    
    print("\n" + "="*50)
    print("NEXT STEPS:")
    print("="*50)
    print("1. Train the model:")
    print(f"   yolo detect train data={yaml_path} model=yolov8n.pt epochs=200")
    print("\n2. For better GPU (your friend's PC):")
    print("   yolo detect train data=dataset.yaml model=yolov8x.pt epochs=300 batch=64")