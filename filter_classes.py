import os
import glob
from datetime import datetime

# --- CONFIGURATION ---
DATA_FOLDER = "raw_data"
LOG_DIR = "logs"

# The "Truth" - Map everything to these IDs
CLASS_MAPPING = {
    'caries': 0,
    'gingivitis': 1,
    'tooth discoloration': 2,
    'ulcer': 3,
    'calculus': 4,
    'plaque': 5,
    'hypodontia': 6,
    # Common Misspellings / Alternate names found in your data
    'lessions': 3,  # Fix misspelling to Ulcer
    'xerostomia': 3 # Map rare class to Ulcer (or remove if preferred)
}
# ---------------------

def log_to_file(lines):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(LOG_DIR, "cleaning_log.txt")
    
    with open(log_file, "w") as f:
        f.write(f"=== DATA CLEANING LOG ({timestamp}) ===\n")
        f.write("\n".join(lines))
        f.write("\n======================================\n")
    print(f"Log saved to: {log_file}")

def clean_labels():
    log_buffer = []
    txt_files = glob.glob(os.path.join(DATA_FOLDER, "*.txt"))
    
    log_buffer.append(f"Scanning {len(txt_files)} label files in '{DATA_FOLDER}'...")
    print(f"Scanning {len(txt_files)} files...")
    
    files_updated = 0
    removed_lines = 0
    detected_classes = set()

    for txt_file in txt_files:
        with open(txt_file, "r") as f:
            lines = f.readlines()
        
        new_lines = []
        changed = False
        
        for line in lines:
            parts = line.strip().split()
            if not parts: continue
            
            # The class ID is always the first number
            original_id = int(parts[0])
            
            # NOTE: Since your text files have IDs (0, 1, 2) not names, 
            # we assume the standard Roboflow map if we see raw IDs.
            # This script primarily fixes the mapping if you had a JSON 
            # or if we are validating ranges. 
            
            # Ideally, we trust the file, but we remove "Ghost" classes (IDs > 6)
            if original_id > 6:
                # Ghost class (e.g. 15 from a different dataset)
                removed_lines += 1
                changed = True
                continue
                
            new_lines.append(line)
            detected_classes.add(original_id)
        
        if changed:
            with open(txt_file, "w") as f:
                f.writelines(new_lines)
            files_updated += 1

    log_buffer.append(f"Classes detected in dataset: {sorted(list(detected_classes))}")
    log_buffer.append(f"Files updated (cleaned): {files_updated}")
    log_buffer.append(f"Irrelevant labels removed: {removed_lines}")
    
    print(f"Cleaning complete. Updated {files_updated} files.")
    
    log_to_file(log_buffer)

if __name__ == "__main__":
    clean_labels()