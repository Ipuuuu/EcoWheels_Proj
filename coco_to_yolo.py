# simple_coco_yolo.py
import json
import shutil
import yaml
import random
from pathlib import Path

# Config
dataset = Path("/home/immaculatapatrickumoh/Documents/EcoWheels_Proj/Org_dataset")
output = Path("/home/immaculatapatrickumoh/Documents/EcoWheels_Proj/yolo_taco")

print("🚀 Simple COCO to YOLO Converter")

# Find annotations file
ann_files = list(dataset.glob("*.json"))
if not ann_files:
    print("❌ No JSON files found!")
    exit()

ann_file = ann_files[0]
print(f"📄 Using: {ann_file.name}")

# Load data
with open(ann_file) as f:
    data = json.load(f)

print(f"📊 Found: {len(data['images'])} images, {len(data['annotations'])} annotations")

# Create mapping
cats = sorted(data['categories'], key=lambda x: x['id'])
cat_map = {c['id']: i for i, c in enumerate(cats)}

# Create directories
for split in ['train', 'val', 'test']:
    (output / split / 'images').mkdir(parents=True, exist_ok=True)
    (output / split / 'labels').mkdir(parents=True, exist_ok=True)

# Group annotations
anns_by_img = {}
for ann in data['annotations']:
    anns_by_img.setdefault(ann['image_id'], []).append(ann)

# Process each image
for img in data['images']:
    filename = img['file_name']
    
    # Find image in batch folder
    if '_' in filename:
        parts = filename.split('_')
        if len(parts) >= 3:
            batch = f"batch_{parts[1]}"
            src = dataset / batch / filename
            
            if src.exists():
                # Choose split
                r = random.random()
                split = 'train' if r < 0.7 else 'val' if r < 0.9 else 'test'
                
                # Copy image
                shutil.copy2(src, output / split / 'images' / filename)
                
                # Create label if annotations exist
                if img['id'] in anns_by_img:
                    label_file = output / split / 'labels' / f"{Path(filename).stem}.txt"
                    with open(label_file, 'w') as f:
                        for ann in anns_by_img[img['id']]:
                            x, y, w, h = ann['bbox']
                            xc = (x + w/2) / img['width']
                            yc = (y + h/2) / img['height']
                            wn = w / img['width']
                            hn = h / img['height']
                            cls = cat_map[ann['category_id']]
                            f.write(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

# Create data.yaml
with open(output / "data.yaml", 'w') as f:
    yaml.dump({
        'path': str(output.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(cats),
        'names': [c['name'] for c in cats]
    }, f)

print(f"\n✅ Done! YOLO dataset at: {output}")
print(f"🎯 Train with: yolo train data={output}/data.yaml model=yolov8s.pt epochs=100")