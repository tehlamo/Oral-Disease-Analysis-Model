import os
import random
import logging

# CONFIGURATION
# ---------------------------------------------------------
RAW_DATA_DIR = "./raw_data"
TRAIN_LIST = "train_list.txt"
VAL_LIST = "val_list.txt"
TEST_LIST = "test_list.txt"
TRAIN_SPLIT = 0.64  # 64% for training
VAL_SPLIT = 0.16    # 16% for validation
# Remaining 20% for testing
LOG_FILE = "logs/dataset_creation_log.txt"

# SETUP LOGGING
# ---------------------------------------------------------
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

def main():
    logging.info("--- STARTING DATASET CREATION ---")
    
    all_images = []
    
    try:
        # Walk through directory
        if not os.path.exists(RAW_DATA_DIR):
            logging.error(f"Directory not found: {RAW_DATA_DIR}")
            return

        for root, dirs, files in os.walk(RAW_DATA_DIR):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    # Create linux path
                    full_path = os.path.join(root, file).replace("\\", "/")
                    all_images.append(full_path)

        if len(all_images) == 0:
            logging.warning("No images found! Check your raw_data folder.")
            return

        # Randomize image order to ensure unbiased data distribution across splits
        random.shuffle(all_images)
        logging.info(f"Found {len(all_images)} valid images.")

        # Stratified split: 64% training, 16% validation, 20% test
        train_idx = int(len(all_images) * TRAIN_SPLIT)
        val_idx = train_idx + int(len(all_images) * VAL_SPLIT)
        
        train_images = all_images[:train_idx]
        val_images = all_images[train_idx:val_idx]
        test_images = all_images[val_idx:]

        logging.info(f"Train set: {len(train_images)} images ({len(train_images)/len(all_images)*100:.1f}%)")
        logging.info(f"Validation set: {len(val_images)} images ({len(val_images)/len(all_images)*100:.1f}%)")
        logging.info(f"Test set: {len(test_images)} images ({len(test_images)/len(all_images)*100:.1f}%)")

        # Write train list
        with open(TRAIN_LIST, 'w') as f:
            for img_path in train_images:
                f.write(f"{img_path}\n")
        logging.info(f"Successfully saved train list to {TRAIN_LIST}")

        # Write validation list
        with open(VAL_LIST, 'w') as f:
            for img_path in val_images:
                f.write(f"{img_path}\n")
        logging.info(f"Successfully saved validation list to {VAL_LIST}")

        # Write test list
        with open(TEST_LIST, 'w') as f:
            for img_path in test_images:
                f.write(f"{img_path}\n")
        logging.info(f"Successfully saved test list to {TEST_LIST}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise e

    logging.info("--- DATASET CREATION COMPLETE ---")

if __name__ == "__main__":
    main()