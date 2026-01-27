import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
import logging
import sys
import multiprocessing

# Import the proper dataset class with YOLO parsing
from load_datasets import DentalDataset

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
VAL_LIST = "val_list.txt"
EARLY_STOPPING_PATIENCE = 10  # Stop if validation loss doesn't improve for 10 epochs
LR_SCHEDULER_PATIENCE = 5     # Reduce LR if validation loss doesn't improve for 5 epochs

# Disease class ordering (matches filter_classes.py)
# Class 0 = background (Faster R-CNN requirement)
# Classes 1-7 = diseases in order
# Note: lesions and xerostomia are mapped to class 3 (ulcer) in filter_classes.py
CLASS_ORDER = {
    0: 'caries',
    1: 'gingivitis', 
    2: 'tooth discoloration',
    3: 'ulcer',
    4: 'calculus',
    5: 'plaque',
    6: 'hypodontia'
}

# OPTIMIZATION SETTINGS
# ---------------------------------------------------------
NUM_WORKERS = 4  # Parallel data loading workers for improved throughput 

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
# Wrapper around DentalDataset to ensure proper disease class ordering
class OralDiseaseDataset(DentalDataset):
    """
    Custom dataset class that extends DentalDataset.
    Uses the proper YOLO annotation parsing and ensures disease class ordering.
    """
    def __init__(self, list_file, transforms=None):
        # Initialize parent class which handles YOLO -> Pascal VOC conversion
        super().__init__(list_file, transforms)
        # The parent class already handles:
        # - Reading YOLO format annotations
        # - Converting to Pascal VOC format (xmin, ymin, xmax, ymax)
        # - Mapping class_id from annotation file to model class (class_id + 1)
        #   where 0 = background, 1-7 = disease classes

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

    # 1. Prepare Training Data
    dataset = OralDiseaseDataset(TRAIN_LIST, transforms=ToTensor())
    data_loader = DataLoader(
        dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=NUM_WORKERS,     # Parallel data loading for improved throughput
        pin_memory=True,             # Enables faster GPU memory transfer
        collate_fn=lambda x: tuple(zip(*x))  # Custom collate function for object detection format
    )
    
    logging.info(f"Loaded {len(dataset)} images for training.")

    # 2. Setup Model
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)
    
    # Initialize optimizer with L2 regularization
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    
    # Adaptive learning rate scheduler: reduces LR by 50% when validation loss plateaus
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_SCHEDULER_PATIENCE
    )

    # 2b. Prepare Validation Data
    val_dataset = OralDiseaseDataset(VAL_LIST, transforms=ToTensor())
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # No shuffling for validation to ensure consistent evaluation
        num_workers=NUM_WORKERS,
        pin_memory=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    logging.info(f"Loaded {len(val_dataset)} images for validation.")

    # Initialize tracking variables for model selection and early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_epoch = 0

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

        avg_train_loss = epoch_loss / len(data_loader)
        
        # Validation Loop: Evaluate model performance on held-out validation set
        model.train()  # Keep in train mode to get loss_dict (Faster R-CNN returns losses when targets provided)
        val_loss = 0
        with torch.no_grad():  # Disable gradient computation for efficiency
            for images, targets in val_loader:
                try:
                    images = list(img.to(DEVICE) for img in images)
                    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                    
                    loss_dict = model(images, targets)
                    # loss_dict is a dictionary, sum all loss components
                    losses = sum(loss for loss in loss_dict.values())
                    val_loss += losses.item()
                except Exception as e:
                    logging.error(f"Error in Validation Batch: {str(e)}")
                    continue
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Update learning rate based on validation loss plateau detection
        lr_scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        logging.info(f"Epoch: {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
        
        # Model selection: Save checkpoint when validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            
            best_model_path = os.path.join(SAVE_DIR, "oral_model_v1_best.pth")
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved! Val Loss: {avg_val_loss:.4f}")
        else:
            epochs_without_improvement += 1
        
        # Periodic checkpoint saving for recovery and analysis
        if (epoch + 1) % 5 == 0 or (epoch + 1) == NUM_EPOCHS:
            save_path = os.path.join(SAVE_DIR, f"oral_model_v1_epoch_{epoch}.pth")
            torch.save(model.state_dict(), save_path)
            logging.info(f"Checkpoint saved: {save_path}")
        
        # Early stopping: Prevent overfitting by halting training when validation loss plateaus
        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            logging.info(f"Early stopping triggered! No improvement for {EARLY_STOPPING_PATIENCE} epochs.")
            logging.info(f"Best model was at epoch {best_epoch} with validation loss: {best_val_loss:.4f}")
            break

    logging.info("--- TRAINING COMPLETE ---")
    logging.info(f"Best model: epoch {best_epoch} with validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    main()