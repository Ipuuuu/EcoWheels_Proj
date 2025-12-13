# material_based_merger.py
import json
from pathlib import Path
import yaml
import shutil
from collections import Counter

# ========== CONFIGURATION ==========
DATASET_PATH = Path("/home/immaculatapatrickumoh/Documents/EcoWheels_Proj/yolo_taco")
ORIGINAL_YAML = DATASET_PATH / "data.yaml"
MERGED_DATASET_PATH = DATASET_PATH.parent / "yolo_taco_material_merged"
# ===================================

def get_material_based_groups():
    """Group classes by material type (better for trash classification)"""
    
    # Based on TacoTrashDataset categories
    material_groups = {
        # PLASTIC items
        'plastic': [
            4,   # Other plastic bottle
            5,   # Clear plastic bottle
            7,   # Plastic bottle cap
            21,  # Disposable plastic cup
            22,  # Foam cup (Styrofoam)
            24,  # Other plastic cup
            27,  # Plastic lid
            29,  # Other plastic
            36,  # Plastic film
            37,  # Six pack rings (plastic)
            39,  # Other plastic wrapper
            40,  # Single-use carrier bag (plastic)
            41,  # Polypropylene bag (plastic)
            42,  # Crisp packet (plastic)
            44,  # Tupperware (plastic)
            45,  # Disposable food container (plastic)
            46,  # Foam food container (plastic/foam)
            47,  # Other plastic container
            48,  # Plastic glooves
            49,  # Plastic utensils
            55,  # Plastic straw
            57,  # Styrofoam piece
            
        ],
        
        # GLASS items
        'glass': [
            6,   # Glass bottle
            9,   # Broken glass
            23,  # Glass cup
            26,  # Glass jar
        ],
        
        # METAL items
        'metal': [
            0,   # Aluminium foil
            8,   # Metal bottle cap
            10,  # Food Can
            11,  # Aerosol
            12,  # Drink can
            28,  # Metal lid
            50,  # Pop tab
            52,  # Scrap metal
        ],
        
        # CARDBOARD items
        'cardboard': [
            14,  # Other carton
            15,  # Egg carton
            16,  # Drink carton
            17,  # Corrugated carton
            18,  # Meal carton
            19,  # Pizza box
        ],

         # PAPER items
        'paper': [
            20,  # Paper cup
            30,  # Magazine paper
            31,  # Tissues
            32,  # Wrapping paper
            33,  # Normal paper
            34,  # Paper bag
            35,  # Plastified paper bag
            56,  # Paper straw
        ],
        
        # SPECIAL items (keep separate due to importance)
        'special': [
            1,   # Battery (hazardous, needs separate recycling)
            2,   # Aluminium blister pack (medicine packaging)
            3,   # Carded blister pack
            13,  # Toilet tube
            25,  # Food waste (organic)
            38,  # Garbage bag
            43,  # Spread tub
            51,  # Rope & strings
            53,  # Shoe
            54,  # Squeezable tube
            
        ],
        
        # MISC/UNLABELED
        'misc': [
            58,  # Unlabeled litter
        ],
        
        # CIGARETTE (very common)
        'cigarette': [
            59,  # Cigarette
        ]
    }
    
    return material_groups

def analyze_all_splits():
    """Analyze class distribution across ALL splits"""
    
    print("📊 Analyzing class distribution across all splits...")
    
    # Your provided distributions
    train_dist = {59: 503, 36: 333, 58: 321, 29: 206, 39: 191, 5: 176, 12: 152, 7: 133, 9: 130, 
                  55: 98, 57: 79, 14: 76, 6: 75, 21: 73, 50: 71, 8: 62, 27: 58, 17: 55, 33: 51, 
                  0: 49, 40: 43, 20: 42, 16: 35, 4: 33, 31: 31, 10: 29, 42: 27, 38: 25, 45: 24, 
                  34: 21, 18: 21, 51: 18, 49: 18, 15: 9, 22: 9, 46: 9, 52: 8, 28: 8, 25: 7, 
                  11: 7, 43: 7, 54: 6, 2: 6, 53: 5, 47: 5, 30: 5, 32: 5, 37: 4, 26: 4, 44: 3, 
                  56: 3, 48: 3, 41: 3, 23: 3, 19: 2, 13: 2, 1: 2, 24: 1}
    
    val_dist = {58: 162, 59: 113, 36: 83, 5: 77, 7: 52, 12: 52, 55: 52, 39: 52, 29: 44, 
                21: 27, 6: 24, 57: 22, 20: 20, 33: 18, 50: 16, 27: 15, 8: 14, 52: 12, 
                49: 12, 14: 12, 45: 11, 40: 11, 51: 10, 31: 10, 42: 9, 0: 9, 4: 9, 18: 8, 
                9: 7, 30: 7, 46: 6, 17: 6, 16: 5, 32: 5, 34: 5, 22: 4, 10: 3, 23: 3, 11: 3, 
                38: 3, 43: 2, 28: 2, 26: 2, 25: 1, 53: 1, 15: 1, 37: 1, 54: 1, 44: 1, 56: 1, 
                19: 1, 3: 1}
    
    test_dist = {59: 51, 36: 35, 58: 34, 5: 32, 12: 25, 7: 24, 29: 23, 39: 17, 33: 13, 
                 50: 12, 57: 11, 4: 8, 49: 7, 55: 7, 40: 7, 20: 5, 16: 5, 6: 5, 14: 5, 
                 21: 4, 8: 4, 27: 4, 0: 4, 42: 3, 45: 3, 13: 3, 38: 3, 17: 3, 32: 2, 
                 10: 2, 34: 1, 24: 1, 18: 1, 48: 1, 15: 1, 51: 1, 53: 1, 47: 1, 9: 1, 31: 1}
    
    # Combine all splits
    total_dist = Counter()
    for dist in [train_dist, val_dist, test_dist]:
        total_dist.update(dist)
    
    # Convert to array for all 60 classes
    total_counts = [total_dist.get(i, 0) for i in range(60)]
    
    return total_counts, total_dist

def create_material_based_merge(total_counts):
    """Create merge mapping based on MATERIAL types"""
    
    print(f"\n🔀 Creating material-based merge mapping...")
    
    material_groups = get_material_based_groups()
    
    # Load original names
    with open(ORIGINAL_YAML, 'r') as f:
        original_config = yaml.safe_load(f)
    original_names = original_config['names']
    
    # Start assigning new IDs
    merge_map = {}
    new_id_counter = 0
    
    print(f"\n📦 Material Groups:")
    
    # Process each material group
    for material, class_ids in material_groups.items():
        # Calculate total samples in this group
        group_total = sum(total_counts[cid] for cid in class_ids)
        
        print(f"  {material.upper()}: {len(class_ids)} classes, {group_total} total samples")
        
        if group_total == 0:
            continue
        
        # Assign all classes in this group to the same new ID
        for class_id in class_ids:
            merge_map[class_id] = new_id_counter
        
        new_id_counter += 1
    
    # Check for any unassigned classes
    unassigned = [i for i in range(60) if i not in merge_map]
    if unassigned:
        print(f"\n⚠️  {len(unassigned)} unassigned classes: {unassigned}")
        for class_id in unassigned:
            merge_map[class_id] = new_id_counter
        new_id_counter += 1
    
    num_final_classes = len(set(merge_map.values()))
    
    print(f"\n📊 Merge summary:")
    print(f"  Original: 60 classes")
    print(f"  Merged: {num_final_classes} material-based classes")
    
    # Create material-based names
    material_names = []
    for material in material_groups.keys():
        if material == 'plastic':
            material_names.append("Plastic")
        elif material == 'glass':
            material_names.append("Glass")
        elif material == 'metal':
            material_names.append("Metal")
        elif material == 'cardboard':
            material_names.append("Cardboard")
        elif material == 'paper':
            material_names.append("Paper")
        elif material == 'special':
            material_names.append("Special_Waste")
        elif material == 'misc':
            material_names.append("Unlabeled_Litter")
        elif material == 'cigarette':
            material_names.append("Cigarette")
    
    return merge_map, num_final_classes, material_names

def simple_material_merge():
    """Even simpler: Merge into 6 main material categories"""
    
    print("\n🎯 SIMPLE MATERIAL MERGE (6 categories)")
    print("=" * 60)
    
    # Super simple material mapping
    material_map = {
        'Plastic': [4, 5, 7, 21, 22, 24, 27, 29, 36, 37, 39, 40, 41, 42, 
                    44, 45, 46, 47, 48, 49, 55, 57], 
        'Glass': [6, 8, 9, 23, 26],
        'Metal': [0, 10, 11, 12, 28, 50, 52],
        'Cardboard': [14, 15, 16, 17, 18, 19],
        'Paper': [20, 30, 31, 32, 33, 34, 35, 56],
        'Special_Waste': [1, 2, 3, 13, 25, 38, 43, 51, 53, 54],
        'Cigarette': [59],
        'Unlabeled': [58]
    }
    
    # Create merge mapping
    merge_map = {}
    for new_id, (material_name, class_ids) in enumerate(material_map.items()):
        for class_id in class_ids:
            merge_map[class_id] = new_id
    
    # Check for any missing
    for i in range(60):
        if i not in merge_map:
            print(f"⚠️  Class {i} not in any material group!")
            merge_map[i] = len(material_map)  # Add to "Other"
    
    material_names = list(material_map.keys())
    
    print(f"\n✅ Merged 60 classes → {len(material_names)} material categories:")
    for i, name in enumerate(material_names):
        print(f"  {i}: {name}")
    
    return merge_map, len(material_names), material_names

def merge_dataset_simple():
    """Simple material-based merge"""
    
    print("=" * 60)
    print("🔄 MATERIAL-BASED CLASS MERGER")
    print("=" * 60)
    
    # Get simple material merge
    merge_map, num_merged_classes, material_names = simple_material_merge()
    
    # Load original names for reporting
    with open(ORIGINAL_YAML, 'r') as f:
        original_config = yaml.safe_load(f)
    original_names = original_config['names']
    
    # Create new dataset directory
    MERGED_DATASET_PATH.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = ['train', 'val', 'test']
    total_updated = 0
    
    for split in splits:
        print(f"\n📁 Processing {split} split...")
        
        # Create directories
        images_dir = MERGED_DATASET_PATH / split / 'images'
        labels_dir = MERGED_DATASET_PATH / split / 'labels'
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        src_images = DATASET_PATH / split / 'images'
        if src_images.exists():
            image_files = list(src_images.glob("*"))
            for img_file in image_files:
                shutil.copy2(img_file, images_dir / img_file.name)
            print(f"  Copied {len(image_files)} images")
        
        # Update label files
        src_labels = DATASET_PATH / split / 'labels'
        if src_labels.exists():
            label_files = list(src_labels.glob("*.txt"))
            updated = 0
            
            for label_file in label_files:
                new_lines = []
                
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        old_class_id = int(parts[0])
                        new_class_id = merge_map.get(old_class_id, old_class_id)
                        parts[0] = str(new_class_id)
                        new_lines.append(' '.join(parts))
                
                if new_lines:
                    output_file = labels_dir / label_file.name
                    with open(output_file, 'w') as f:
                        f.write('\n'.join(new_lines))
                    updated += 1
            
            print(f"  Updated {updated} label files")
            total_updated += updated
    
    # Create new data.yaml
    new_config = {
        'path': str(MERGED_DATASET_PATH.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': num_merged_classes,
        'names': material_names
    }
    
    new_yaml = MERGED_DATASET_PATH / "data.yaml"
    with open(new_yaml, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)
    
    print(f"\n" + "=" * 60)
    print("🎉 MATERIAL MERGE COMPLETE!")
    print("=" * 60)
    print(f"\n📁 New dataset: {MERGED_DATASET_PATH}")
    print(f"📊 60 classes → {num_merged_classes} material categories")
    
    # Create simple report
    report = [
        "=" * 60,
        "MATERIAL-BASED CLASS MERGING",
        "=" * 60,
        f"\nOriginal: 60 specific trash items",
        f"Merged: {num_merged_classes} material categories",
        f"\nMaterial Categories:",
        "-" * 40
    ]
    
    for i, material in enumerate(material_names):
        # Find original classes in this material
        original_in_material = []
        for old_id, new_id in merge_map.items():
            if new_id == i:
                original_in_material.append(f"{old_id}: {original_names[old_id]}")
        
        report.append(f"\n{i}: {material}")
        report.append(f"  Contains {len(original_in_material)} original classes:")
        for item in original_in_material[:5]:  # Show first 5
            report.append(f"    - {item}")
        if len(original_in_material) > 5:
            report.append(f"    ... and {len(original_in_material)-5} more")
    
    report_file = MERGED_DATASET_PATH / "material_merge_report.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"📄 Report: {report_file}")
    print(f"\n🚀 Train with: yolo train data={new_yaml} model=yolov8x.pt epochs=300")
    
    return MERGED_DATASET_PATH

if __name__ == "__main__":
    # Run the simple material merge
    merged_path = merge_dataset_simple()