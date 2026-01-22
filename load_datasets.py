import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class DentalDataset(Dataset):
    """
    A custom PyTorch Dataset for loading dental images and annotations.
    
    This class is responsible for:
    1. Reading image paths from a provided text file list (e.g., train_fold_0.txt).
    2. Loading the corresponding YOLO label file.
    3. Converting normalized YOLO coordinates (center_x, center_y, width, height)
       to Pascal VOC pixel coordinates (xmin, ymin, xmax, ymax).
    4. Returning the image and target dictionary as tensors.
    """

    def __init__(self, list_path, transforms = None):
        # Read the list of image paths for the specific fold
        with open(list_path, 'r') as f:
            self.image_paths = f.read().splitlines()
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        Load image and corresponding annotations for a given index.
        
        Returns:
            image: PIL Image converted to tensor format
            target: Dictionary containing bounding boxes, labels, and image_id
        """
        # Load image and ensure RGB format (3 channels)
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # Locate corresponding annotation file (YOLO format)
        label_path = os.path.splitext(img_path)[0] + ".txt"

        boxes = []
        labels = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()

                    # Parse YOLO format: class_id center_x center_y width height (all normalized 0-1)
                    class_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])

                    # Convert normalized YOLO coordinates to absolute pixel coordinates (Pascal VOC format)
                    x_min = (cx - bw / 2) * w
                    y_min = (cy - bh / 2) * h
                    x_max = (cx + bw / 2) * w
                    y_max = (cy + bh / 2) * h

                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(class_id + 1)  # Offset by 1: 0=background, 1+=disease classes

        # Convert to tensors: empty tensors for images without annotations (healthy/background)
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype = torch.float32)
            labels = torch.as_tensor(labels, dtype = torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype = torch.float32)
            labels = torch.zeros((0,), dtype = torch.int64)

        # Format target dictionary for Faster R-CNN compatibility
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        # Apply transformations (typically converts PIL Image to tensor)
        if self.transforms:
            img = self.transforms(img)

        return img, target
    
    def __len__(self):
        return len(self.image_paths)