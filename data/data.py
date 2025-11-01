import os
import shutil
from pathlib import Path
import csv

def rename_images_and_create_csv(base_directory, output_directory=None, start_number=1, create_backup=True):
    """
    Renames images to sequential numbers (1.jpg, 2.jpg, etc.) and creates a CSV mapping.
    
    Args:
        base_directory: Path to the main directory containing condition folders
        output_directory: Path where renamed images will be saved (None = create in base_directory)
        start_number: Starting number for image numbering
        create_backup: If True, keeps original files in backup folder
    """
    
    base_path = Path(base_directory)
    
    if not base_path.exists():
        print(f"Error: Directory '{base_directory}' does not exist!")
        return
    
    # Setup output directory
    if output_directory:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path = base_path / "renamed_images"
        output_path.mkdir(exist_ok=True)
    
    # Create backup directory if needed
    if create_backup:
        backup_path = base_path / "original_backup"
        backup_path.mkdir(exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # Collect all image data
    image_data = []
    current_number = start_number
    
    # Get all folders and sort them
    folders = [f for f in sorted(base_path.iterdir()) 
               if f.is_dir() and not f.name.startswith('.') and f.name not in ['renamed_images', 'original_backup']]
    
    print(f"Found {len(folders)} disease folders")
    print("=" * 70)
    
    # Process each subdirectory
    for folder in folders:
        disease_label = folder.name
        print(f"\nProcessing: {disease_label}")
        
        # Get all image files
        image_files = [f for f in sorted(folder.iterdir()) 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"  No images found")
            continue
        
        # Backup original folder if requested
        if create_backup:
            backup_disease_path = backup_path / disease_label
            if not backup_disease_path.exists():
                shutil.copytree(folder, backup_disease_path)
        
        # Rename each image
        for image_file in image_files:
            # Get file extension
            extension = image_file.suffix.lower()
            
            # Create new numbered filename
            new_name = f"{current_number}{extension}"
            new_path = output_path / new_name
            
            # Copy to output directory with new name
            shutil.copy2(image_file, new_path)
            
            # Record the mapping
            image_data.append({
                'image_number': current_number,
                'new_filename': new_name,
                'disease_label': disease_label,
                'original_filename': image_file.name,
                'original_folder': disease_label
            })
            
            print(f"  {image_file.name} -> {new_name} (Label: {disease_label})")
            current_number += 1
        
        print(f"  Images processed: {len(image_files)}")
    
    # Write to CSV
    if image_data:
        csv_path = output_path / "image_labels.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['image_number', 'new_filename', 
                                                   'disease_label', 'original_filename', 
                                                   'original_folder'])
            writer.writeheader()
            writer.writerows(image_data)
        
        print("\n" + "=" * 70)
        print("✓ SUCCESS!")
        print("=" * 70)
        print(f"\nRenamed images saved to: {output_path}")
        print(f"CSV mapping saved to: {csv_path}")
        if create_backup:
            print(f"Original images backed up to: {backup_path}")
        
        # Print summary
        print(f"\nTotal images processed: {len(image_data)}")
        print(f"Image numbers: {start_number} to {current_number - 1}")
        
        print("\nBreakdown by disease:")
        disease_counts = {}
        for item in image_data:
            label = item['disease_label']
            disease_counts[label] = disease_counts.get(label, 0) + 1
        
        for disease, count in sorted(disease_counts.items()):
            print(f"  {disease}: {count} images")
        
        print("\n" + "=" * 70)
        
    else:
        print("\nNo images found to process!")


if __name__ == "__main__":
    print("=" * 70)
    print("EYE CONDITION IMAGE RENAMER + CSV GENERATOR")
    print("=" * 70)
    print("\nThis script will:")
    print("  1. Rename all images to sequential numbers (1.jpg, 2.jpg, etc.)")
    print("  2. Create a CSV mapping each number to its disease label")
    print("  3. Keep your original files safe in a backup folder")
    
    # Get directory path from user
    base_dir = input("\nEnter the path to your eye images directory: ").strip()
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory '{base_dir}' does not exist!")
        exit()
    
    # Ask about output location
    print("\nWhere should renamed images be saved?")
    print("  1. Create 'renamed_images' folder in the same directory (recommended)")
    print("  2. Specify a custom output directory")
    
    choice = input("Enter choice (1-2): ").strip()
    
    output_dir = None
    if choice == "2":
        output_dir = input("Enter output directory path: ").strip()
    
    # Get starting number
    start_num = input("\nStarting number (press Enter for 1): ").strip()
    start_num = int(start_num) if start_num else 1
    
    # Ask about backup
    backup = input("Create backup of original files? (yes/no, default=yes): ").strip().lower()
    create_backup = backup != "no"
    
    print("\n" + "=" * 70)
    print("Processing...")
    print("=" * 70)
    
    # Run the renaming and CSV creation
    rename_images_and_create_csv(base_dir, output_dir, start_num, create_backup)

    #/Users/rudranshagrawal/Desktop/MAIS25/data/