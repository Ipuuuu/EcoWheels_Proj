# rename_images.py - UPDATED VERSION
import os
from pathlib import Path
import shutil

# ========== CONFIGURATION ==========
DATASET_PATH = Path("/home/immaculatapatrickumoh/Documents/EcoWheels_Proj/Org_dataset")
BACKUP_FIRST = True  # Set to True to create backup before renaming
# ===================================

def create_backup():
    """Create backup of original dataset"""
    backup_path = DATASET_PATH.parent / "Org_dataset_backup"
    
    if backup_path.exists():
        print(f"⚠️  Backup already exists at: {backup_path}")
        response = input("Overwrite backup? (y/n): ").strip().lower()
        if response != 'y':
            print("Skipping backup...")
            return
    
    print(f"📦 Creating backup to: {backup_path}")
    
    # Remove existing backup if it exists
    if backup_path.exists():
        shutil.rmtree(backup_path)
    
    # Copy entire dataset
    shutil.copytree(DATASET_PATH, backup_path)
    print(f"✅ Backup created: {len(list(backup_path.rglob('*.*')))} files")

def standardize_extensions():
    """First, standardize all extensions to lowercase BEFORE renaming"""
    
    print("\n🔄 STANDARDIZING FILE EXTENSIONS")
    print("-" * 40)
    
    total_standardized = 0
    
    for batch_num in range(1, 16):
        batch_dir = DATASET_PATH / f"batch_{batch_num}"
        
        if not batch_dir.exists():
            continue
        
        # Convert uppercase extensions to lowercase
        for uppercase_ext in ['.JPG', '.JPEG', '.PNG']:
            for file_path in batch_dir.glob(f"*{uppercase_ext}"):
                # Create new name with lowercase extension
                new_name = file_path.stem + uppercase_ext.lower()
                new_path = batch_dir / new_name
                
                # Rename the file
                file_path.rename(new_path)
                total_standardized += 1
                
                if total_standardized <= 5:
                    print(f"  Standardized: {file_path.name} → {new_name}")
    
    if total_standardized > 0:
        print(f"\n✅ Standardized {total_standardized} file extensions")
    else:
        print("✅ All extensions already standardized")

def rename_all_images():
    """Rename all image files to include batch number with consistent extensions"""
    
    print("\n🔄 RENAMING ALL IMAGES")
    print("=" * 60)
    
    total_renamed = 0
    
    for batch_num in range(1, 16):
        batch_dir = DATASET_PATH / f"batch_{batch_num}"
        
        if not batch_dir.exists():
            print(f"❌ Batch {batch_num} not found: {batch_dir}")
            continue
        
        print(f"\n📁 Processing {batch_dir.name}:")
        
        # Get all image files (now all lowercase extensions)
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(sorted(batch_dir.glob(f"*{ext}")))
        
        print(f"  Found {len(image_files)} images")
        
        # Rename each file
        batch_renamed = 0
        for img_path in image_files:
            # Get original name
            original_name = img_path.name
            name_without_ext = img_path.stem
            extension = img_path.suffix.lower()  # Already lowercase
            
            # Create new name: batch_X_originalname.ext (lowercase)
            new_name = f"batch_{batch_num}_{name_without_ext}{extension}"
            new_path = batch_dir / new_name
            
            # Check if already renamed
            if original_name == new_name:
                print(f"  ⚠️  Already renamed: {original_name}")
                continue
            
            # Rename the file
            try:
                img_path.rename(new_path)
                batch_renamed += 1
                total_renamed += 1
                
                if batch_renamed <= 3:  # Show first 3 renames per batch
                    print(f"  ✅ {original_name} → {new_name}")
                    
            except Exception as e:
                print(f"  ❌ Failed to rename {original_name}: {e}")
        
        print(f"  Renamed {batch_renamed} files in this batch")
    
    print(f"\n" + "=" * 60)
    print(f"🎯 TOTAL RENAMED: {total_renamed} images")
    print("=" * 60)
    
    return total_renamed

def verify_renaming():
    """Verify that files were renamed correctly and have consistent extensions"""
    
    print("\n🔍 VERIFICATION")
    print("-" * 40)
    
    # Check a few batches
    sample_batches = [1, 5, 10, 15]
    
    for batch_num in sample_batches:
        batch_dir = DATASET_PATH / f"batch_{batch_num}"
        
        if not batch_dir.exists():
            continue
        
        print(f"\nChecking {batch_dir.name}:")
        
        # Get first 5 files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(batch_dir.glob(f"*{ext}"))
        
        # Sort and show first 5
        image_files = sorted(image_files)[:5]
        
        for img_path in image_files:
            filename = img_path.name
            
            # Check 1: filename starts with batch_X_
            correct_prefix = filename.startswith(f"batch_{batch_num}_")
            
            # Check 2: extension is lowercase
            correct_extension = img_path.suffix in ['.jpg', '.jpeg', '.png']
            
            if correct_prefix and correct_extension:
                print(f"  ✅ {filename} (correct)")
            else:
                issues = []
                if not correct_prefix:
                    issues.append("missing batch prefix")
                if not correct_extension:
                    issues.append(f"bad extension: {img_path.suffix}")
                print(f"  ❌ {filename} ({', '.join(issues)})")
        
        # Count file types
        jpg_count = len(list(batch_dir.glob("*.jpg")))
        JPG_count = len(list(batch_dir.glob("*.JPG")))  # Should be 0
        jpeg_count = len(list(batch_dir.glob("*.jpeg")))
        JPEG_count = len(list(batch_dir.glob("*.JPEG")))  # Should be 0
        png_count = len(list(batch_dir.glob("*.png")))
        PNG_count = len(list(batch_dir.glob("*.PNG")))  # Should be 0
        
        if JPG_count > 0 or JPEG_count > 0 or PNG_count > 0:
            print(f"  ⚠️  Uppercase extensions found: .JPG={JPG_count}, .JPEG={JPEG_count}, .PNG={PNG_count}")

def create_rename_report(total_renamed):
    """Create a report of the renaming process"""
    
    print("\n📋 CREATING RENAME REPORT")
    print("-" * 40)
    
    report_lines = [
        "=" * 60,
        "IMAGE RENAMING REPORT",
        "=" * 60,
        f"\nDataset: {DATASET_PATH}",
        f"Total images renamed: {total_renamed}",
        f"Timestamp: {os.popen('date').read().strip()}",
        "\nBatch Summary:",
        "-" * 40
    ]
    
    # Count files per batch
    for batch_num in range(1, 16):
        batch_dir = DATASET_PATH / f"batch_{batch_num}"
        
        if batch_dir.exists():
            # Count all image files
            image_count = 0
            for ext in ['.jpg', '.jpeg', '.png']:
                image_count += len(list(batch_dir.glob(f"*{ext}")))
            
            report_lines.append(f"batch_{batch_num}: {image_count} images")
    
    # Save report
    report_file = DATASET_PATH / "rename_report.txt"
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"📄 Report saved to: {report_file}")
    
    return report_file

if __name__ == "__main__":
    print("=" * 60)
    print("🔄 COMPLETE IMAGE RENAMING PROCESS")
    print("=" * 60)
    
    # Step 0: Backup if requested
    if BACKUP_FIRST:
        create_backup()
    
    # Step 1: Standardize extensions to lowercase
    standardize_extensions()
    
    # Step 2: Rename all images
    total_renamed = rename_all_images()
    
    # Step 3: Verify
    verify_renaming()
    
    # Step 4: Create report
    report_file = create_rename_report(total_renamed)
    
    print("\n" + "=" * 60)
    print("🎉 RENAMING PROCESS COMPLETE!")
    print("=" * 60)
    print("\nNext step: Run 'python update_annotations.py'")