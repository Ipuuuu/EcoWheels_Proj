# update_annotations.py - UPDATED VERSION
import json
import os
from pathlib import Path
from collections import Counter

# ========== CONFIGURATION ==========
DATASET_PATH = Path("/home/immaculatapatrickumoh/Documents/EcoWheels_Proj/Org_dataset")
ANNOTATIONS_FILE = DATASET_PATH / "annotations.json"
# ===================================

def update_annotations():
    """Update annotations.json to match renamed images with lowercase extensions"""
    
    print("📝 UPDATING ANNOTATIONS")
    print("=" * 60)
    
    # Load original annotations
    if not ANNOTATIONS_FILE.exists():
        print(f"❌ Annotations file not found: {ANNOTATIONS_FILE}")
        return None
    
    with open(ANNOTATIONS_FILE, 'r') as f:
        data = json.load(f)
    
    print(f"📊 Original dataset stats:")
    print(f"  Images: {len(data['images']):,}")
    print(f"  Annotations: {len(data['annotations']):,}")
    print(f"  Categories: {len(data['categories'])}")
    print(f"  Sample categories: {[cat['name'] for cat in data['categories'][:5]]}")
    
    print(f"\n🔄 Updating image paths...")
    print("-" * 40)
    
    updated_count = 0
    errors = []
    
    for idx, img in enumerate(data['images']):
        old_path = img['file_name']
        
        # Expected format: "batch_X/filename.ext"
        if '/' in old_path:
            parts = old_path.split('/')
            
            if len(parts) == 2:
                batch_folder, old_filename = parts
                
                # Extract batch number from folder name
                if batch_folder.startswith('batch_'):
                    try:
                        batch_num = int(batch_folder.replace('batch_', ''))
                        
                        # Split filename and extension
                        if '.' in old_filename:
                            name_part, ext_part = old_filename.rsplit('.', 1)
                            # Convert extension to lowercase
                            extension = f".{ext_part.lower()}"
                        else:
                            name_part = old_filename
                            extension = ""
                        
                        # Create new filename: batch_X_filename.ext (lowercase)
                        new_filename = f"batch_{batch_num}_{name_part}{extension}"
                        
                        # Update the path in annotations
                        img['file_name'] = new_filename
                        updated_count += 1
                        
                        # Show first few updates
                        if updated_count <= 5:
                            print(f"  {old_path} → {new_filename}")
                            
                    except ValueError as e:
                        errors.append(f"Could not parse batch number in '{batch_folder}': {e}")
                else:
                    errors.append(f"Folder doesn't start with 'batch_': {batch_folder}")
            else:
                errors.append(f"Unexpected path format (not 2 parts): {old_path}")
        else:
            errors.append(f"No slash in path (already flat?): {old_path}")
    
    print(f"\n✅ Updated {updated_count:,} image paths")
    
    if errors:
        print(f"\n⚠️  {len(errors)} errors encountered:")
        for error in errors[:5]:
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    
    # Save updated annotations
    updated_file = DATASET_PATH / "annotations_updated.json"
    with open(updated_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n💾 Saved to: {updated_file}")
    
    # Verify the update
    verification_results = verify_annotations(data)
    
    return data, updated_file, verification_results

def verify_annotations(data):
    """Verify that annotations match actual files"""
    
    print(f"\n🔍 VERIFYING UPDATED ANNOTATIONS")
    print("-" * 40)
    
    # Check 1: Unique filenames
    filenames = [img['file_name'] for img in data['images']]
    duplicate_counts = Counter(filenames)
    duplicates = {name: count for name, count in duplicate_counts.items() if count > 1}
    
    if duplicates:
        print(f"⚠️  Found {len(duplicates)} duplicate filenames:")
        for name, count in list(duplicates.items())[:3]:
            print(f"  - {name}: {count} times")
    else:
        print(f"✅ All {len(filenames):,} filenames are unique")
    
    # Check 2: Extension consistency
    print(f"\n📁 Checking extension consistency:")
    
    extensions_counter = Counter()
    for filename in filenames:
        if '.' in filename:
            ext = filename.split('.')[-1].lower()
            extensions_counter[ext] += 1
    
    print(f"  File extensions:")
    for ext, count in extensions_counter.most_common():
        print(f"    .{ext}: {count:,} files")
    
    # Check 3: File existence (sample)
    print(f"\n🔍 Checking file existence (sample of 20 files):")
    
    import random
    sample_size = min(20, len(data['images']))
    sample_indices = random.sample(range(len(data['images'])), sample_size)
    
    existing_count = 0
    missing_files = []
    
    for idx in sample_indices:
        img = data['images'][idx]
        filename = img['file_name']
        
        # Extract batch number from filename
        if '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 3:
                batch_num = parts[1]  # "batch_1_000000.jpg" → "1"
                batch_folder = f"batch_{batch_num}"
                
                # Check if file exists
                file_path = DATASET_PATH / batch_folder / filename
                
                if file_path.exists():
                    existing_count += 1
                else:
                    missing_files.append(filename)
            else:
                missing_files.append(filename)
        else:
            missing_files.append(filename)
    
    print(f"  Files found: {existing_count}/{sample_size}")
    
    if missing_files:
        print(f"  Missing files (first 5):")
        for missing in missing_files[:5]:
            print(f"    - {missing}")
    
    # Check 4: Batch distribution
    print(f"\n📊 Batch distribution:")
    
    batch_counts = {}
    for img in data['images']:
        filename = img['file_name']
        if '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 2:
                batch_key = f"batch_{parts[1]}"
                batch_counts[batch_key] = batch_counts.get(batch_key, 0) + 1
    
    # Sort by batch number
    for batch in sorted(batch_counts.keys(), key=lambda x: int(x.split('_')[1])):
        count = batch_counts[batch]
        print(f"  {batch}: {count:,} images")
    
    return {
        'total_images': len(data['images']),
        'unique_filenames': len(set(filenames)),
        'duplicates': len(duplicates),
        'sample_found': existing_count,
        'sample_total': sample_size,
        'batch_counts': batch_counts
    }

def create_annotation_report(data, updated_file_path, verification_results):
    """Create detailed annotation report"""
    
    print(f"\n📋 CREATING ANNOTATION REPORT")
    print("=" * 60)
    
    report_lines = [
        "=" * 60,
        "ANNOTATION UPDATE REPORT",
        "=" * 60,
        f"\nDataset: {DATASET_PATH}",
        f"Updated annotations file: {updated_file_path.name}",
        f"Timestamp: {os.popen('date').read().strip()}",
        "",
        "DATASET STATISTICS",
        "-" * 40,
        f"Total images: {verification_results['total_images']:,}",
        f"Total annotations: {len(data['annotations']):,}",
        f"Categories: {len(data['categories'])}",
        f"Unique filenames: {verification_results['unique_filenames']:,}",
        f"Duplicate filenames: {verification_results['duplicates']:,}",
        f"Sample check: {verification_results['sample_found']}/{verification_results['sample_total']} files found",
        "",
        "BATCH DISTRIBUTION",
        "-" * 40
    ]
    
    # Add batch counts
    for batch in sorted(verification_results['batch_counts'].keys(), 
                       key=lambda x: int(x.split('_')[1])):
        count = verification_results['batch_counts'][batch]
        report_lines.append(f"{batch}: {count:,} images")
    
    # Add categories
    report_lines.extend([
        "",
        "CATEGORIES",
        "-" * 40
    ])
    
    for cat in data['categories']:
        report_lines.append(f"{cat['id']}: {cat['name']}")
    
    # Add annotation statistics
    report_lines.extend([
        "",
        "ANNOTATION STATISTICS",
        "-" * 40
    ])
    
    # Count annotations per category
    category_counts = Counter()
    for ann in data['annotations']:
        category_counts[ann['category_id']] += 1
    
    report_lines.append(f"Annotations per category:")
    for cat_id, count in sorted(category_counts.items()):
        cat_name = next((cat['name'] for cat in data['categories'] if cat['id'] == cat_id), "Unknown")
        report_lines.append(f"  {cat_name} (ID {cat_id}): {count:,}")
    
    # Save report
    report_file = DATASET_PATH / "annotation_update_report.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 Report saved to: {report_file}")
    
    return report_file

if __name__ == "__main__":
    results = update_annotations()
    
    if results:
        data, updated_file, verification_results = results
        
        # Create detailed report
        report_file = create_annotation_report(data, updated_file, verification_results)
        
        print("\n" + "=" * 60)
        print("🎉 ANNOTATION UPDATE COMPLETE!")
        print("=" * 60)
        
        success_rate = (verification_results['sample_found'] / verification_results['sample_total']) * 100
        
        if success_rate >= 90:
            print(f"\n✅ READY FOR UPLOAD! ({success_rate:.1f}% files verified)")
            print(f"\nUpload command:")
            print(f'roboflow import -w ipuuuu -p ecowheels "{updated_file}"')
        else:
            print(f"\n⚠️  CHECK REQUIRED! Only {success_rate:.1f}% files verified")
            print("Check the verification report above for missing files.")
        
        print(f"\nNext step: Run 'python verify_dataset.py' for complete verification")