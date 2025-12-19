import os
import shutil
import zipfile
import random

# --- CONFIGURATION ---
# Define paths for source archives and destination directory
MULTI_ZIP_PATH = os.path.join("downloads", "multi.zip")
CARIES_ZIP_PATH = os.path.join("downloads", "caries.zip")
HEALTHY_ZIP_PATH = os.path.join("downloads", "healthy.zip")

DEST_DIR = "raw_data"
# ---------------------

def find_matching_image(txt_path, search_root):
    """
    Locates the corresponding image file for a given text annotation file.
    Searches in the same directory, parallel 'images' directories, and recursively.
    """
    basename = os.path.splitext(os.path.basename(txt_path))[0]
    parent_dir = os.path.dirname(txt_path)
    
    # 1. Check for image in the same directory
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
        path = os.path.join(parent_dir, basename + ext)
        if os.path.exists(path): return path
            
    # 2. Check for standard dataset structure (labels/ vs images/)
    if "labels" in parent_dir:
        img_dir = parent_dir.replace("labels", "images")
        for ext in [".jpg", ".jpeg", ".png", ".JPG", ".PNG"]:
            path = os.path.join(img_dir, basename + ext)
            if os.path.exists(path): return path

    # 3. Perform recursive search in root directory
    for root, dirs, files in os.walk(search_root):
        for file in files:
            if os.path.splitext(file)[0] == basename and file.lower().endswith(('.jpg', '.jpeg', '.png')):
                return os.path.join(root, file)
    return None

def process_zip(zip_path, prefix, limit=None, is_healthy=False):
    print(f"Processing archive: {zip_path}")
    
    if not os.path.exists(zip_path):
        print(f"Error: File not found at {zip_path}")
        return 0

    # Create temporary extraction directory
    temp_dir = f"temp_{prefix}"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_dir)
    except Exception as e:
        print(f"Error extracting archive: {e}")
        return 0
    
    # CASE A: Process Healthy Data (Images only, discard labels)
    if is_healthy:
        count = 0
        valid_images = []
        
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG')):
                    valid_images.append(os.path.join(root, file))
        
        if limit: random.shuffle(valid_images)
        
        for src_path in valid_images:
            if limit and count >= limit: break
            
            new_name = f"{prefix}_{os.path.basename(src_path)}"
            try:
                shutil.copy2(src_path, os.path.join(DEST_DIR, new_name))
                count += 1
            except Exception: pass
            
        shutil.rmtree(temp_dir)
        return count

    # CASE B: Process Annotated Data (Image and Text pairs)
    all_txt = []
    for root, dirs, files in os.walk(temp_dir):
        for f in files:
            # Filter for annotation files only
            if f.endswith(".txt") and "classes" not in f and "data" not in f:
                all_txt.append(os.path.join(root, f))
    
    print(f"Labels found in archive: {len(all_txt)}")
    if limit: random.shuffle(all_txt)
    
    count = 0
    for txt in all_txt:
        if limit and count >= limit: break
        
        img = find_matching_image(txt, temp_dir)
        if img:
            # Rename files to prevent naming conflicts
            new_txt = f"{prefix}_{os.path.basename(txt)}"
            new_img = f"{prefix}_{os.path.basename(img)}"
            try:
                shutil.copy2(txt, os.path.join(DEST_DIR, new_txt))
                shutil.copy2(img, os.path.join(DEST_DIR, new_img))
                count += 1
            except Exception: pass
            
    shutil.rmtree(temp_dir)
    return count

def rebuild_dataset():
    # Clear existing data directory
    if os.path.exists(DEST_DIR): shutil.rmtree(DEST_DIR)
    os.makedirs(DEST_DIR, exist_ok=True)
    print(f"Directory cleared and recreated: {DEST_DIR}")

    # 1. Import Multi-Disease Dataset (Base)
    c_multi = process_zip(MULTI_ZIP_PATH, "multi")
    print(f"Multi-Disease images imported: {c_multi}")

    # 2. Import Caries Dataset (Balanced)
    # Limit caries samples to 1.2x the count of multi-disease samples to prevent class imbalance
    limit_caries = int(c_multi * 1.2) if c_multi > 0 else 2000
    print(f"Applying balance limit to Caries dataset: {limit_caries}")
    c_caries = process_zip(CARIES_ZIP_PATH, "zenodo", limit=limit_caries)
    print(f"Caries images imported: {c_caries}")

    # 3. Import Healthy Dataset (Balanced)
    # Limit healthy samples to 0.5x the count of multi-disease samples
    limit_healthy = int(c_multi * 0.5) if c_multi > 0 else 1000
    print(f"Importing Healthy/Background images (Limit: {limit_healthy})...")
    c_healthy = process_zip(HEALTHY_ZIP_PATH, "healthy", limit=limit_healthy, is_healthy=True)
    print(f"Healthy images imported: {c_healthy}")

    print("-" * 40)
    print(f"Dataset construction complete. Total images: {c_multi + c_caries + c_healthy}")

if __name__ == "__main__":
    rebuild_dataset()