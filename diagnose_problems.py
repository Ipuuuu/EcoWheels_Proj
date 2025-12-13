# diagnose_problems.py
from ultralytics import YOLO
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def diagnose_low_mAP():
    """Diagnose why mAP is only 11%"""
    
    # 1. Check dataset configuration
    data_yaml = Path("/home/immaculatapatrickumoh/Documents/EcoWheels_Proj/yolo_taco_material_merged/data.yaml")
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    print("🔍 DIAGNOSING LOW mAP (11%)")
    print("=" * 60)
    
    # 2. Load model and check predictions
    model_path = "runs/detect/trash_material_v2/weights/best.pt"
    if Path(model_path).exists():
        model = YOLO(model_path)
        
        # Validate on a few images
        print("\n📊 Checking predictions on validation set...")
        
        # Get validation directory
        val_dir = Path(config['path']) / "val/images"
        val_images = list(val_dir.glob("*"))[:5]
        
        for img_path in val_images:
            results = model(img_path, conf=0.25)
            
            # Check if predictions make sense
            for result in results:
                print(f"\nImage: {img_path.name}")
                print(f"  Detected {len(result.boxes)} objects")
                
                if result.boxes:
                    classes = result.boxes.cls.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    
                    for cls, conf in zip(classes, confs):
                        cls_name = config['names'][int(cls)]
                        print(f"    - {cls_name}: {conf:.2f}")
    
    # 3. Check label files
    print("\n📝 Checking label files...")
    train_labels = list((Path(config['path']) / "train/labels").glob("*.txt"))
    
    if train_labels:
        sample_label = train_labels[0]
        with open(sample_label, 'r') as f:
            content = f.read()
        
        print(f"Sample label ({sample_label.name}):")
        print(f"  Content: {content[:100]}...")
        
        # Parse first annotation
        lines = content.strip().split('\n')
        if lines and lines[0]:
            parts = lines[0].split()
            if len(parts) >= 5:
                cls_id, xc, yc, w, h = parts[:5]
                print(f"  Class: {cls_id}, Center: ({xc}, {yc}), Size: ({w}, {h})")
                
                # Check if coordinates are normalized (0-1)
                try:
                    xc, yc, w, h = float(xc), float(yc), float(w), float(h)
                    if xc < 0 or xc > 1 or yc < 0 or yc > 1:
                        print(f"  ❌ ERROR: Coordinates not normalized! (should be 0-1)")
                    else:
                        print(f"  ✅ Coordinates normalized correctly")
                except:
                    print(f"  ❌ ERROR: Could not parse coordinates")

# Run diagnosis
diagnose_low_mAP()