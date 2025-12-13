import os
import random
import shutil
from pathlib import Path
from collections import Counter
import yaml

INPUT_DIR = "/home/immaculatapatrickumoh/Documents/EcoWheels_Proj/balanced_data"
OUTPUT_DIR = "/home/immaculatapatrickumoh/Documents/EcoWheels_Proj/balanced_final"
TARGET_PER_CLASS = 800  # Target images per class

print("Starting downsampling...")
print(f"Input: {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Target: {TARGET_PER_CLASS} images per class\n")

# Create output directories
output_train_img = Path(OUTPUT_DIR) / "train" / "images"
output_train_lbl = Path(OUTPUT_DIR) / "train" / "labels"
output_val_img = Path(OUTPUT_DIR) / "val" / "images"
output_val_lbl = Path(OUTPUT_DIR) / "val" / "labels"

for d in [output_train_img, output_train_lbl, output_val_img, output_val_lbl]:
    d.mkdir(parents=True, exist_ok=True)

# ====== 1. COPY VALIDATION SET (UNCHANGED) ======
print("Copying validation set...")
val_img_src = Path(INPUT_DIR) / "val" / "images"
val_lbl_src = Path(INPUT_DIR) / "val" / "labels"

if val_img_src.exists() and val_lbl_src.exists():
    val_copied = 0
    for img_file in val_img_src.glob("*"):
        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            label_file = val_lbl_src / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy(img_file, output_val_img / img_file.name)
                shutil.copy(label_file, output_val_lbl / label_file.name)
                val_copied += 1
    print(f"  Copied {val_copied} validation images")
else:
    print("  WARNING: Validation folder not found!")

# ====== 2. ANALYZE TRAINING DISTRIBUTION ======
print("\nAnalyzing current distribution...")
train_img_src = Path(INPUT_DIR) / "train" / "images"
train_lbl_src = Path(INPUT_DIR) / "train" / "labels"

# Count IMAGES (not annotations) per class
class_image_counts = Counter()
image_class_map = {}  # filename -> set of classes

image_files = list(train_img_src.glob("*"))
print(f"Found {len(image_files)} training images")

for img_file in image_files:
    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        label_file = train_lbl_src / f"{img_file.stem}.txt"
        
        if not label_file.exists():
            continue
        
        # Read classes in this image
        classes_in_image = set()
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        class_id = int(float(line.split()[0]))
                        classes_in_image.add(class_id)
        except:
            continue
        
        if classes_in_image:
            image_class_map[img_file.stem] = classes_in_image
            for class_id in classes_in_image:
                class_image_counts[class_id] += 1

print("\nCurrent IMAGE distribution per class:")
for class_id in sorted(class_image_counts.keys()):
    count = class_image_counts[class_id]
    print(f"  Class {class_id}: {count} images")

# ====== 3. CALCULATE KEEP PROBABILITIES ======
print("\nCalculating keep probabilities...")
keep_probabilities = {}
for class_id in class_image_counts:
    current = class_image_counts[class_id]
    if current > TARGET_PER_CLASS:
        prob = TARGET_PER_CLASS / current
        keep_probabilities[class_id] = prob
        print(f"  Class {class_id}: {current} → {TARGET_PER_CLASS} (keep {prob:.1%})")
    else:
        keep_probabilities[class_id] = 1.0
        print(f"  Class {class_id}: {current} → {current} (keep 100%)")

# ====== 4. DOWNSAMPLE TRAINING SET ======
print("\nDownsampling training set...")
images_kept = 0
images_skipped = 0

for img_file in train_img_src.glob("*"):
    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
        label_file = train_lbl_src / f"{img_file.stem}.txt"
        
        if not label_file.exists():
            continue
        
        # Get classes in this image
        classes_in_image = image_class_map.get(img_file.stem, set())
        
        if not classes_in_image:
            continue
        
        # Calculate keep probability
        # Keep image if ANY class in it needs to be kept
        keep_prob = 1.0
        for class_id in classes_in_image:
            if class_id in keep_probabilities:
                keep_prob = min(keep_prob, keep_probabilities[class_id])
        
        # Randomly decide to keep
        if random.random() <= keep_prob:
            shutil.copy(img_file, output_train_img / img_file.name)
            shutil.copy(label_file, output_train_lbl / label_file.name)
            images_kept += 1
        else:
            images_skipped += 1

print(f"\nDownsampling results:")
print(f"  Images kept: {images_kept}")
print(f"  Images skipped: {images_skipped}")
print(f"  Keep rate: {images_kept/(images_kept+images_skipped):.1%}")

# ====== 5. VERIFY NEW DISTRIBUTION ======
print("\nVerifying new distribution...")
new_class_counts = Counter()
for label_file in output_train_lbl.glob("*.txt"):
    try:
        with open(label_file, 'r') as f:
            classes_in_file = set()
            for line in f:
                line = line.strip()
                if line:
                    class_id = int(float(line.split()[0]))
                    classes_in_file.add(class_id)
            
            for class_id in classes_in_file:
                new_class_counts[class_id] += 1
    except:
        continue

print("\nNew IMAGE distribution per class:")
for class_id in sorted(new_class_counts.keys()):
    print(f"  Class {class_id}: {new_class_counts[class_id]} images")

# ====== 6. CREATE DATASET.YAML ======
print("\nCreating dataset.yaml...")
dataset_yaml = {
    'path': str(Path(OUTPUT_DIR).absolute()),
    'train': 'train/images',
    'val': 'val/images',
    'names': {0: "Cardboard", 1: "Glass", 2: "Metal", 3: "Mixed Waste", 
              4: "Organic Waste", 5: "Paper", 6: "Plastic", 7: "Textiles"},
    'nc': 8
}

yaml_path = Path(OUTPUT_DIR) / "dataset.yaml"
with open(yaml_path, 'w') as f:
    yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

print(f"Created {yaml_path}")

print(f"\n✅ Downsampling complete!")
print(f"📁 Original: {INPUT_DIR} (unchanged)")
print(f"📁 New balanced: {OUTPUT_DIR}")
print(f"📊 Target: ~{TARGET_PER_CLASS} images per class")
print(f"\nTrain with: yolo detect train data={yaml_path} model=yolov8n.pt")
