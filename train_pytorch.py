import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from load_datasets import DentalDataset
from torch.utils.data import DataLoader
import os
from datetime import datetime

# --- HYPERPARAMETERS ---
NUM_CLASSES = 7  # 6 diseases + 1 background
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.005
CURRENT_FOLD = 0 
LOG_FILE = "logs/training_log.txt"
# -----------------------

def log_to_file(msg):
    """Appends a message to the permanent training log."""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] {msg}\n")
    # Also print to console so it shows up in the SLURM .out file
    print(msg)

def get_model(num_classes):
    print("Loading Faster R-CNN model...")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # Detect if GPU is available (Fail Fast)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        log_to_file(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError("ERROR: No GPU detected! Check SLURM script.")

    log_to_file(f"--- STARTING TRAINING SESSION ---")
    log_to_file(f"Configuration: Fold {CURRENT_FOLD} | Batch Size {BATCH_SIZE} | LR {LEARNING_RATE}")

    train_list = f"data_splits/train_fold_{CURRENT_FOLD}.txt"
    val_list = f"data_splits/val_fold_{CURRENT_FOLD}.txt"

    if not os.path.exists(train_list):
        log_to_file(f"Error: {train_list} not found.")
        return

    # Prepare Data
    data_transform = transforms.Compose([transforms.ToTensor()])
    log_to_file("Initializing DataLoaders...")
    
    train_dataset = DentalDataset(train_list, transforms=data_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Initialize Model
    model = get_model(NUM_CLASSES)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)

    log_to_file(f"Initiating training loop for {NUM_EPOCHS} epochs...")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        
        for i, (images, targets) in enumerate(train_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            
            if i % 50 == 0:
                print(f"Epoch: {epoch} | Step: {i} | Batch Loss: {losses.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        
        # LOG THE CRITICAL METRIC
        log_to_file(f"Epoch {epoch} Complete. Average Loss: {avg_loss:.4f}")
        
        save_path = f"dental_rcnn_fold{CURRENT_FOLD}_epoch{epoch}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Checkpoint saved: {save_path}")

    log_to_file("--- TRAINING COMPLETE ---")

if __name__ == "__main__":
    main()