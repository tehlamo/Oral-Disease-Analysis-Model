import os
import glob
import numpy as np
from sklearn.model_selection import KFold

# --- CONFIGURATION ---
DATA_FOLDER = "raw_data"
OUTPUT_FOLDER = "data_splits"
NUM_FOLDS = 5
RANDOM_SEED = 42
# ---------------------

def create_folds():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    print("Scanning directory for images...")
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
    
    # USE A SET TO AUTOMATICALLY REMOVE DUPLICATES
    unique_paths = set()
    
    for ext in extensions:
        found = glob.glob(os.path.join(DATA_FOLDER, ext))
        # Normalize paths to handle Windows case-insensitivity
        for p in found:
            unique_paths.add(os.path.abspath(p))
    
    # Convert back to list and sort for consistency
    images = sorted(list(unique_paths))
    images = np.array(images)

    # SANITY CHECK: The number here MUST match your build.py count (~6593)
    if len(images) == 0:
        print("❌ Error: No images found.")
        return

    print(f"Total unique images found: {len(images)}")
    
    # DOUBLE CHECK
    if len(images) > 7000:
        print("⚠️ WARNING: Count is still high. Check for duplicate files in raw_data!")
    
    print(f"Generating {NUM_FOLDS}-Fold splits...")

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

        print(f"Fold {fold_index} created: {len(train_imgs)} Train, {len(val_imgs)} Val.")

    print("-" * 40)
    print(f"K-Fold generation complete. Files saved to: {OUTPUT_FOLDER}/")

if __name__ == "__main__":
    create_folds()