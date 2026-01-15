import os
import random
import logging

# CONFIGURATION
# ---------------------------------------------------------
RAW_DATA_DIR = "./raw_data"
OUTPUT_LIST = "train_list.txt"
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

        # Shuffle for randomness
        random.shuffle(all_images)
        logging.info(f"Found {len(all_images)} valid images.")

        # Write to file
        with open(OUTPUT_LIST, 'w') as f:
            for img_path in all_images:
                f.write(f"{img_path}\n")

        logging.info(f"Successfully saved image list to {OUTPUT_LIST}")

    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise e

    logging.info("--- DATASET CREATION COMPLETE ---")

if __name__ == "__main__":
    main()