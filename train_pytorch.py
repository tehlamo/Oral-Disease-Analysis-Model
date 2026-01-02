import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from load_datasets import DentalDataset
from torch.utils.data import DataLoader
import os

# --- HYPERPARAMETERS ---
# See "Part 3: Documentation" below for details on modifying these values.
NUM_CLASSES = 7  # 6 disease classes + 1 background class
BATCH_SIZE = 4   # Number of images processed per step
NUM_EPOCHS = 10  # Number of passes through the entire dataset
LEARNING_RATE = 0.005 # Step size for the optimizer
CURRENT_FOLD = 0 # The specific K-Fold split to train on (0-4)
# -----------------------

def get_model(num_classes):
    """
    Loads the Faster R-CNN model with a ResNet50 backbone.
    Replaces the pre-trained head with a new one matching our number of classes.
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights = "DEFAULT")

    # Replace the predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def collate_fn(batch):
    """
    Custom collate function to handle batches of images with varying numbers of boxes.
    """
    return tuple(zip(*batch))

def main():
    # Detect if GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        # CRITICAL: Force a crash if no GPU is found.
        # This prevents the script from silently running on CPU for days.
        raise RuntimeError("No GPU detected.")
    
    # Define paths to the split files
    train_list = f"data_splits/train_fold_{CURRENT_FOLD}.txt"
    val_list = f"data_splits/val_fold_{CURRENT_FOLD}.txt"

    if not os.path.exists(train_list):
        print(f"Error: {train_list} not found.")
        return
    
    # Data transformations
    data_transform = transforms.Compose([transforms.ToTensor()])

    # Initialize Datasets and DataLoaders
    train_dataset = DentalDataset(train_list, transforms = data_transform)
    val_dataset = DentalDataset(val_list, transforms = data_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        collate_fn = collate_fn
    )

    model = get_model(NUM_CLASSES)
    model.to(device)

    # Configure Optimizer (Stochastic Gradient Descent)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr = LEARNING_RATE, momentum = 0.9, weight_decay = 0.0005)

    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0

        for i, (images, targets) in enumerate(train_loader):
            # Move images/targets to GPU
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

            if i % 50 == 0:
                print(f"Epoch: {epoch} | Step: {i} | Batch Loss: {losses.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} complete. Average Loss: {avg_loss:.4f}")

        # Save model checkpoint
        save_path = f"dental_rcnn_fold{CURRENT_FOLD}_epoch{epoch}.pth"
        torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()