import os
import glob
import numpy as np
from sklearn.model_selection import KFold
from datetime import datetime

# --- CONFIGURATION ---
DATA_FOLDER = "raw_data"
OUTPUT_FOLDER = "data_splits"
LOG_DIR = "logs"
NUM_FOLDS = 5
RANDOM_SEED = 42
# ---------------------

def log_to_file(lines):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file = os.path.join(LOG_DIR, "splitting_log.txt")
    
    with open(log_file, "w") as f:
        f.write(f"=== K-FOLD SPLIT LOG ({timestamp}) ===\n")
        f.write("\n".join(lines))
        f.write("\n======================================\n")
    print(f"Log saved to: {log_file}")

def create_folds():
    log_buffer = []
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("Scanning directory for images...")
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    
    # DEDUPLICATION LOGIC
    unique_paths = set()
    for ext in extensions:
        found = glob.glob(os.path.join(DATA_FOLDER, ext))
        for p in found:
            unique_paths.add(os.path.abspath(p))
            
    images = sorted(list(unique_paths))
    images = np.array(images)

    if len(images) == 0:
        print("Error: No images found.")
        return

    log_buffer.append(f"Configuration: Folds={NUM_FOLDS}, Seed={RANDOM_SEED}")
    log_buffer.append(f"Total Unique Images: {len(images)}")
    print(f"Total unique images found: {len(images)}")
    
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    for fold_index, (train_idx, val_idx) in enumerate(kf.split(images)):
        train_imgs = images[train_idx]
        val_imgs = images[val_idx]

        train_file = os.path.join(OUTPUT_FOLDER, f"train_fold_{fold_index}.txt")
        val_file = os.path.join(OUTPUT_FOLDER, f"val_fold_{fold_index}.txt")

        with open(train_file, "w") as f:
            f.write("\n".join(train_imgs))
        with open(val_file, "w") as f:
            f.write("\n".join(val_imgs))

        msg = f"Fold {fold_index}: Train={len(train_imgs)}, Val={len(val_imgs)}"
        log_buffer.append(msg)
        print(msg)

    log_buffer.append(f"Split files saved to: {OUTPUT_FOLDER}/")
    print("-" * 40)
    print(f"K-Fold generation complete.")
    
    log_to_file(log_buffer)

if __name__ == "__main__":
    create_folds()