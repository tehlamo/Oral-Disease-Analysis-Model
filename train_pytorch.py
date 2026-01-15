import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import logging
import sys
import multiprocessing

# CONFIGURATION
# ---------------------------------------------------------
BATCH_SIZE = 4
NUM_EPOCHS = 50
LEARNING_RATE = 0.005
NUM_CLASSES = 20  # Safety buffer
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
SAVE_DIR = "production_models"
LOG_FILE = "logs/training_run_log.txt"
TRAIN_LIST = "train_list.txt"

# OPTIMIZATION SETTINGS
# ---------------------------------------------------------
# We use 4 workers to pre-load data.
NUM_WORKERS = 4 

# SETUP LOGGING
# ---------------------------------------------------------
if not os.path.exists("logs"):
    os.makedirs("logs")

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# CUSTOM DATASET CLASS
# ---------------------------------------------------------
class OralDiseaseDataset(Dataset):
    def __init__(self, list_file, transforms=None):
        self.transforms = transforms
        self.image_paths = []
        
        try:
            with open(list_file, 'r') as f:
                self.image_paths = [line.strip() for line in f.readlines()]
        except FileNotFoundError:
            logging.error(f"Could not find {list_file}. Did you run create_dataset.py?")
            sys.exit(1)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            logging.error(f"Failed to load image: {img_path} | Error: {e}")
            return torch.zeros((3, 200, 200)), {}

        # Placeholder Target (Replace with actual XML parsing if needed)
        boxes = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32)
        labels = torch.tensor([1], dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}
        
        if self.transforms is not None:
            img = self.transforms(img)
            
        return img, target

    def __len__(self):
        return len(self.image_paths)

# MODEL SETUP
# ---------------------------------------------------------
def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# MAIN TRAINING LOOP
# ---------------------------------------------------------
def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    logging.info(f"--- STARTING PRODUCTION TRAINING ---")
    logging.info(f"Device: {DEVICE} | Epochs: {NUM_EPOCHS} | Workers: {NUM_WORKERS}")

    # 1. Prepare Data
    dataset = OralDiseaseDataset(TRAIN_LIST, transforms=torchvision.transforms.ToTensor())
    
    # OPTIMIZATION: num_workers=4 speeds up data loading significantly
    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,     # <--- THE SPEED BOOST
        pin_memory=True,             # <--- FASTER GPU TRANSFER
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    logging.info(f"Loaded {len(dataset)} images for training.")

    # 2. Setup Model
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    # 3. Training Loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0
        
        for i, (images, targets) in enumerate(data_loader):
            try:
                images = list(image.to(DEVICE) for image in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                epoch_loss += losses.item()
            except Exception as e:
                logging.error(f"Error in Batch {i}: {str(e)}")
                continue

        avg_loss = epoch_loss / len(data_loader)
        logging.info(f"Epoch: {epoch} | Average Loss: {avg_loss:.4f}")

        # Save Checkpoint
        if (epoch + 1) % 5 == 0 or (epoch + 1) == NUM_EPOCHS:
            save_path = os.path.join(SAVE_DIR, f"oral_model_v1_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            logging.info(f"Checkpoint saved: {save_path}")

    logging.info("--- TRAINING COMPLETE ---")

if __name__ == "__main__":
    main()