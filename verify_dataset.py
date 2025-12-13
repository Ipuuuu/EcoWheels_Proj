# verify_dataset.py - UPDATED VERSION
import json
import os
from pathlib import Path
from collections import Counter
import random

# ========== CONFIGURATION ==========
DATASET_PATH = Path("/home/immaculatapatrickumoh/Documents/EcoWheels_Proj/Org_dataset")
# ===================================

def load_annotations():
    """Load the updated annotations file"""
    
    print("📄 LOADING ANNOTATIONS")
    print("-" * 40)
    
    # Try to find the latest annotations file
    annotation_files = [
        DATASET_PATH / "annotations_final.json",
        DATASET_PATH / "annotations_updated.json",
        DATASET_PATH / "annotations.json"
    ]
    
    for ann_file in annotation_files:
        if ann_file.exists():
            print(f"Found: {ann_file.name}")
            with open(ann_file, 'r') as f:
                data = json.load(f)
            return data, ann_file
    
    print("❌ No annotation file found!")
    return None, None

def check_filename_format(data):
    """Check that all filenames follow the correct format"""
    
    print(f"\n📝 CHECKING FILENAME FORMAT")
    print("-" * 40)
    
    correct_format = 0
    incorrect_format = []
    
    for img in data['images']:
        filename = img['file_name']
        
        # Should be: batch_X_000000.jpg (lowercase extension)
        if filename.startswith('batch_') and '_' in filename:
            parts = filename.split('_')
            
            # Check 1: Has at least 3 parts (batch, number, filename)
            if len(parts) >= 3:
                # Check 2: Second part is a number (batch number)
                if parts[1].isdigit():
                    # Check 3: Extension is lowercase
                    if '.' in filename:
                        ext = filename.split('.')[-1]
                        if ext == ext.lower():
                            correct_format += 1
                        else:
                            incorrect_format.append((filename, "uppercase extension"))
                    else:
                        correct_format += 1
                else:
                    incorrect_format.append((filename, "batch not a number"))
            else:
                incorrect_format.append((filename, "not enough parts"))
        else:
            incorrect_format.append((filename, "doesn't start with batch_ or missing _"))
    
    print(f"✅ Correct format: {correct_format:,}")
    print(f"❌ Incorrect format: {len(incorrect_format):,}")
    
    if incorrect_format:
        print(f"\nFirst 5 incorrect filenames:")
        for filename, issue in incorrect_format[:5]:
            print(f"  - {filename} ({issue})")
    
    return correct_format, len(incorrect_format)

def check_file_existence(data, sample_size=100):
    """Check if files actually exist on disk"""
    
    print(f"\n📁 CHECKING FILE EXISTENCE")
    print("-" * 40)
    
    # Use stratified sampling to ensure we check all batches
    images_by_batch = {}
    for img in data['images']:
        filename = img['file_name']
        if '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                batch_num = parts[1]
                if batch_num not in images_by_batch:
                    images_by_batch[batch_num] = []
                images_by_batch[batch_num].append(img)
    
    # Sample from each batch proportionally
    sample_indices = []
    for batch_num, batch_images in images_by_batch.items():
        batch_sample_size = max(1, int(len(batch_images) / len(data['images']) * sample_size))
        if len(batch_images) > batch_sample_size:
            batch_sample = random.sample(range(len(batch_images)), batch_sample_size)
            for idx in batch_sample:
                sample_indices.append((batch_num, idx))
        else:
            # Include all images from small batches
            for idx in range(len(batch_images)):
                sample_indices.append((batch_num, idx))
    
    print(f"Checking {len(sample_indices)} files (stratified sample)...")
    
    existing_files = 0
    missing_files = []
    batch_results = {}
    
    for batch_num, img_idx in sample_indices:
        img = images_by_batch[batch_num][img_idx]
        filename = img['file_name']
        
        batch_folder = f"batch_{batch_num}"
        file_path = DATASET_PATH / batch_folder / filename
        
        if file_path.exists():
            existing_files += 1
            if batch_num not in batch_results:
                batch_results[batch_num] = {'found': 0, 'total': 0}
            batch_results[batch_num]['found'] += 1
        else:
            missing_files.append(filename)
            if batch_num not in batch_results:
                batch_results[batch_num] = {'found': 0, 'total': 0}
        
        if batch_num in batch_results:
            batch_results[batch_num]['total'] += 1
    
    print(f"\n📊 Overall results:")
    print(f"  Files checked: {len(sample_indices):,}")
    print(f"  Files found: {existing_files:,}")
    print(f"  Files missing: {len(missing_files):,}")
    print(f"  Success rate: {(existing_files/len(sample_indices)*100):.1f}%")
    
    if missing_files:
        print(f"\n⚠️  Missing files (first 10):")
        missing_by_batch = Counter()
        for missing in missing_files[:10]:
            print(f"  - {missing}")
            if '_' in missing:
                batch_num = missing.split('_')[1]
                missing_by_batch[batch_num] += 1
        
        if missing_by_batch:
            print(f"\n  Missing files by batch:")
            for batch_num, count in missing_by_batch.most_common():
                print(f"    batch_{batch_num}: {count} missing")
    
    # Show batch-by-batch results
    print(f"\n📈 Batch-by-batch success rates:")
    for batch_num in sorted(batch_results.keys(), key=int):
        stats = batch_results[batch_num]
        if stats['total'] > 0:
            success_rate = (stats['found'] / stats['total']) * 100
            print(f"  batch_{batch_num}: {stats['found']}/{stats['total']} ({success_rate:.1f}%)")
    
    return existing_files, len(sample_indices), missing_files

def check_for_duplicates(data):
    """Check for duplicate filenames"""
    
    print(f"\n🔄 CHECKING FOR DUPLICATES")
    print("-" * 40)
    
    filenames = [img['file_name'] for img in data['images']]
    duplicate_counts = Counter(filenames)
    duplicates = {name: count for name, count in duplicate_counts.items() if count > 1}
    
    if duplicates:
        print(f"⚠️  Found {len(duplicates):,} duplicate filenames:")
        for name, count in list(duplicates.items())[:5]:
            print(f"  - {name}: {count} times")
        
        # Show which batches have duplicates
        duplicate_batches = set()
        for filename in duplicates.keys():
            if '_' in filename:
                batch_num = filename.split('_')[1]
                duplicate_batches.add(batch_num)
        
        if duplicate_batches:
            print(f"\n  Batches with duplicates: {sorted(duplicate_batches)}")
    else:
        print(f"✅ All {len(filenames):,} filenames are unique")
    
    return len(duplicates)

def check_batch_distribution(data):
    """Check image distribution across batches"""
    
    print(f"\n📊 BATCH DISTRIBUTION")
    print("-" * 40)
    
    batch_counts = {}
    for img in data['images']:
        filename = img['file_name']
        if '_' in filename:
            parts = filename.split('_')
            if len(parts) >= 2 and parts[1].isdigit():
                batch_key = f"batch_{parts[1]}"
                batch_counts[batch_key] = batch_counts.get(batch_key, 0) + 1
    
    total_images = sum(batch_counts.values())
    
    print(f"Total images across all batches: {total_images:,}")
    print(f"\nImages per batch:")
    
    # Sort by batch number
    for batch in sorted(batch_counts.keys(), key=lambda x: int(x.split('_')[1])):
        count = batch_counts[batch]
        percentage = (count / total_images) * 100
        print(f"  {batch}: {count:,} images ({percentage:.1f}%)")
    
    return batch_counts

def check_annotations_integrity(data):
    """Check annotations integrity"""
    
    print(f"\n🔍 CHECKING ANNOTATIONS INTEGRITY")
    print("-" * 40)
    
    # Count annotations per image
    annotations_per_image = Counter()
    for ann in data['annotations']:
        annotations_per_image[ann['image_id']] += 1
    
    # Find images with no annotations
    image_ids = {img['id'] for img in data['images']}
    annotated_image_ids = set(annotations_per_image.keys())
    images_without_annotations = image_ids - annotated_image_ids
    
    print(f"Total images: {len(image_ids):,}")
    print(f"Images with annotations: {len(annotated_image_ids):,}")
    print(f"Images without annotations: {len(images_without_annotations):,}")
    
    if images_without_annotations:
        print(f"\n⚠️  {len(images_without_annotations):,} images have no annotations")
    
    # Annotation statistics
    if annotations_per_image:
        avg_annotations = sum(annotations_per_image.values()) / len(annotations_per_image)
        max_annotations = max(annotations_per_image.values())
        min_annotations = min(annotations_per_image.values())
        
        print(f"\n📈 Annotation statistics:")
        print(f"  Average annotations per image: {avg_annotations:.1f}")
        print(f"  Minimum annotations: {min_annotations}")
        print(f"  Maximum annotations: {max_annotations}")
    
    return len(images_without_annotations)

def create_final_report(data, format_results, existence_results, 
                       duplicate_count, batch_counts, missing_annotations):
    """Create final verification report"""
    
    print(f"\n📋 CREATING FINAL VERIFICATION REPORT")
    print("=" * 60)
    
    existing_files, total_checked, missing_files = existence_results
    correct_format, incorrect_format = format_results
    
    success_rate = (existing_files / total_checked) * 100
    format_success_rate = (correct_format / len(data['images'])) * 100
    
    report_lines = [
        "=" * 60,
        "DATASET VERIFICATION REPORT",
        "=" * 60,
        f"\nDataset: {DATASET_PATH}",
        f"Verification date: {os.popen('date').read().strip()}",
        "",
        "OVERALL STATISTICS",
        "-" * 40,
        f"Total images in annotations: {len(data['images']):,}",
        f"Total annotations: {len(data['annotations']):,}",
        f"Categories: {len(data['categories'])}",
        f"Unique filenames: {len(set(img['file_name'] for img in data['images'])):,}",
        f"Duplicate filenames: {duplicate_count:,}",
        f"Images without annotations: {missing_annotations:,}",
        "",
        "VERIFICATION RESULTS",
        "-" * 40,
        f"Filename format correct: {correct_format:,}/{len(data['images']):,} ({format_success_rate:.1f}%)",
        f"Files found on disk: {existing_files:,}/{total_checked:,} ({success_rate:.1f}%)",
        f"Missing files in sample: {len(missing_files):,}",
        "",
        "BATCH DISTRIBUTION",
        "-" * 40
    ]
    
    # Add batch distribution
    total_images = sum(batch_counts.values())
    for batch in sorted(batch_counts.keys(), key=lambda x: int(x.split('_')[1])):
        count = batch_counts[batch]
        percentage = (count / total_images) * 100
        report_lines.append(f"{batch}: {count:,} images ({percentage:.1f}%)")
    
    # Add category list
    report_lines.extend([
        "",
        "CATEGORIES (first 20)",
        "-" * 40
    ])
    
    for cat in data['categories'][:20]:
        report_lines.append(f"{cat['id']}: {cat['name']}")
    
    if len(data['categories']) > 20:
        report_lines.append(f"... and {len(data['categories']) - 20} more categories")
    
    # Save report
    report_file = DATASET_PATH / "final_verification_report.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 Report saved to: {report_file}")
    
    return report_file, success_rate, format_success_rate

def main():
    print("=" * 60)
    print("🔍 COMPLETE DATASET VERIFICATION")
    print("=" * 60)
    
    # Load annotations
    data, ann_file = load_annotations()
    
    if not data:
        print("❌ Failed to load annotations. Exiting.")
        return
    
    print(f"\n📊 Dataset loaded successfully:")
    print(f"  File: {ann_file.name}")
    print(f"  Images: {len(data['images']):,}")
    print(f"  Annotations: {len(data['annotations']):,}")
    print(f"  Categories: {len(data['categories'])}")
    
    # Run all checks
    format_results = check_filename_format(data)
    existence_results = check_file_existence(data, sample_size=100)
    duplicate_count = check_for_duplicates(data)
    batch_counts = check_batch_distribution(data)
    missing_annotations = check_annotations_integrity(data)
    
    # Create final report
    report_file, success_rate, format_success_rate = create_final_report(
        data, format_results, existence_results, duplicate_count, 
        batch_counts, missing_annotations
    )
    
    print("\n" + "=" * 60)
    print("📋 FINAL SUMMARY")
    print("=" * 60)
    
    print(f"\n✅ Files with correct format: {format_success_rate:.1f}%")
    print(f"✅ Files found on disk: {success_rate:.1f}%")
    print(f"✅ Unique filenames: {duplicate_count == 0}")
    
    # Overall assessment
    if success_rate >= 95 and format_success_rate >= 95 and duplicate_count == 0:
        print(f"\n🎉 DATASET IS READY FOR UPLOAD!")
        print(f"\nNext command:")
        print(f'roboflow import -w ipuuuu -p ecowheels "{ann_file}"')
        print(f"\nOr if using web interface:")
        print(f"1. Go to https://app.roboflow.com")
        print(f"2. Create project 'ecowheels'")
        print(f"3. Upload {ann_file.name} as COCO format")
    elif success_rate >= 80:
        print(f"\n⚠️  DATASET HAS MINOR ISSUES ({success_rate:.1f}% success)")
        print("Check the verification report for details.")
        print("You can still upload, but some files may be missing.")
    else:
        print(f"\n❌ DATASET HAS MAJOR ISSUES ({success_rate:.1f}% success)")
        print("Fix the issues before uploading to Roboflow.")
    
    print(f"\n📄 Detailed report: {report_file}")

if __name__ == "__main__":
    main()