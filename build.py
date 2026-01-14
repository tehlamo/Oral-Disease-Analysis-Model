import os
import zipfile
import shutil
import random
import glob
from datetime import datetime

# --- CONFIGURATION ---
DOWNLOADS_DIR = "downloads"
RAW_DATA_DIR = "raw_data"
LOG_DIR = "logs"
TARGET_SIZE_PER_CLASS = 1500  # Cap for diseases
HEALTHY_LIMIT = 500           # Cap for healthy/background
# ---------------------

def log_to_file(lines):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(LOG_DIR, "build_log.txt")
    
    with open(log_file, "w") as f:
        f.write(f"=== DATA BUILD LOG ({timestamp}) ===\n")
        f.write("\n".join(lines))
        f.write("\n======================================\n")
    print(f"📄 Log saved to: {log_file}")

def build_dataset():
    log_buffer = []
    
    # Clean and recreate raw_data
    if os.path.exists(RAW_DATA_DIR):
        shutil.rmtree(RAW_DATA_DIR)
    os.makedirs(RAW_DATA_DIR)
    
    log_buffer.append(f"Action: Recreated '{RAW_DATA_DIR}' directory.")
    print(f"Directory cleared and recreated: {RAW_DATA_DIR}")

    # --- PROCESS MULTI-DISEASE ZIP ---
    multi_zip = os.path.join(DOWNLOADS_DIR, "multi.zip")
    if os.path.exists(multi_zip):
        print(f"Processing archive: {multi_zip}")
        log_buffer.append(f"Source: {multi_zip}")
        
        with zipfile.ZipFile(multi_zip, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            txt_files = [f for f in file_list if f.endswith('.txt') and not f.startswith('__MACOSX')]
            log_buffer.append(f"  - Labels found in archive: {len(txt_files)}")
            print(f"Labels found in archive: {len(txt_files)}")

            count = 0
            for txt_file in txt_files:
                # Extract .txt
                zip_ref.extract(txt_file, RAW_DATA_DIR)
                base_name = os.path.splitext(os.path.basename(txt_file))[0]
                
                # Find and extract matching image
                img_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']
                found_img = False
                for ext in img_extensions:
                    img_name = txt_file.replace('.txt', ext)
                    if img_name in file_list:
                        zip_ref.extract(img_name, RAW_DATA_DIR)
                        found_img = True
                        break
                
                if found_img:
                    count += 1
            
            log_buffer.append(f"  - Images extracted: {count}")
            print(f"Multi-Disease images imported: {count}")

    # --- PROCESS CARIES ZIP (Balanced) ---
    caries_zip = os.path.join(DOWNLOADS_DIR, "caries.zip")
    if os.path.exists(caries_zip):
        print(f"Processing archive: {caries_zip}")
        log_buffer.append(f"Source: {caries_zip}")
        
        with zipfile.ZipFile(caries_zip, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            txt_files = [f for f in all_files if f.endswith('.txt') and "train" in f]
            
            log_buffer.append(f"  - Total labels available: {len(txt_files)}")
            print(f"Labels found in archive: {len(txt_files)}")
            
            # Shuffle and limit
            random.shuffle(txt_files)
            selected_files = txt_files[:TARGET_SIZE_PER_CLASS]
            
            count = 0
            for txt_file in selected_files:
                zip_ref.extract(txt_file, RAW_DATA_DIR)
                
                # Extract corresponding image
                img_extensions = ['.jpg', '.jpeg', '.png']
                for ext in img_extensions:
                    img_name = txt_file.replace('.txt', ext)
                    if img_name in all_files:
                        zip_ref.extract(img_name, RAW_DATA_DIR)
                        count += 1
                        break
            
            log_buffer.append(f"  - Images imported (Capped): {count}")
            print(f"Caries images imported: {count}")

    # --- PROCESS HEALTHY ZIP ---
    healthy_zip = os.path.join(DOWNLOADS_DIR, "healthy.zip")
    if os.path.exists(healthy_zip):
        print(f"Processing archive: {healthy_zip}")
        log_buffer.append(f"Source: {healthy_zip}")
        
        with zipfile.ZipFile(healthy_zip, 'r') as zip_ref:
            all_files = zip_ref.namelist()
            img_files = [f for f in all_files if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            random.shuffle(img_files)
            selected = img_files[:HEALTHY_LIMIT]
            
            count = 0
            for img in selected:
                zip_ref.extract(img, RAW_DATA_DIR)
                count += 1
            
            log_buffer.append(f"  - Healthy images imported: {count}")
            print(f"Healthy images imported: {count}")

    # Final Count
    total_files = len(glob.glob(os.path.join(RAW_DATA_DIR, "*")))
    images = [f for f in glob.glob(os.path.join(RAW_DATA_DIR, "*")) if f.endswith(('.jpg', '.png'))]
    
    summary = f"Dataset construction complete. Total images: {len(images)}"
    log_buffer.append(summary)
    print(summary)
    
    # Write Log
    log_to_file(log_buffer)

if __name__ == "__main__":
    build_dataset()