import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import os
import json
import logging
from collections import defaultdict
import numpy as np

# Import the proper dataset class with YOLO parsing
from load_datasets import DentalDataset

# CONFIGURATION
# ---------------------------------------------------------
MODEL_PATH = "production_models/oral_model_v1_best.pth"  # Path to trained model checkpoint
TEST_LIST = "test_list.txt"  # Test set (held-out 20% of total dataset)
NUM_CLASSES = 20
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
IOU_THRESHOLD = 0.5  # Standard IoU threshold for mAP calculation
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for predictions
OUTPUT_FILE = "logs/evaluation_results.json"
LOG_FILE = "logs/evaluation_log.txt"
# ---------------------------------------------------------

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
# ---------------------------------------------------------

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Calculate intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def calculate_ap_per_class(predictions, ground_truths, num_classes, iou_threshold=0.5):
    """
    Calculate Average Precision (AP) per class using 11-point interpolation method.
    
    Implements standard COCO evaluation metric for object detection.
    """
    ap_per_class = {}
    
    for class_id in range(1, num_classes + 1):  # Skip background (0)
        # Get all predictions and ground truths for this class
        pred_boxes = []
        gt_boxes = []
        
        for pred, gt in zip(predictions, ground_truths):
            # Filter predictions for this class
            pred_mask = pred['labels'] == class_id
            pred_scores = pred['scores'][pred_mask].cpu().numpy()
            pred_boxes_class = pred['boxes'][pred_mask].cpu().numpy()
            
            # Filter ground truths for this class
            gt_mask = gt['labels'] == class_id
            gt_boxes_class = gt['boxes'][gt_mask].cpu().numpy()
            
            # Store with scores for sorting
            for i, box in enumerate(pred_boxes_class):
                pred_boxes.append({
                    'box': box,
                    'score': pred_scores[i],
                    'matched': False
                })
            
            gt_boxes.append({
                'boxes': gt_boxes_class,
                'matched': [False] * len(gt_boxes_class)
            })
        
        if len(pred_boxes) == 0:
            ap_per_class[class_id] = 0.0
            continue
        
        # Sort predictions by confidence score (descending)
        pred_boxes.sort(key=lambda x: x['score'], reverse=True)
        
        # Calculate TP and FP
        tp = []
        fp = []
        num_gt = sum(len(gt['boxes']) for gt in gt_boxes)
        
        for pred_box in pred_boxes:
            best_iou = 0.0
            best_gt_idx = -1
            best_img_idx = -1
            
            # Find best matching ground truth
            for img_idx, gt_img in enumerate(gt_boxes):
                for gt_idx, gt_box in enumerate(gt_img['boxes']):
                    if gt_img['matched'][gt_idx]:
                        continue
                    iou = calculate_iou(pred_box['box'], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                        best_img_idx = img_idx
            
            if best_iou >= iou_threshold:
                tp.append(1)
                fp.append(0)
                gt_boxes[best_img_idx]['matched'][best_gt_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        
        # Calculate precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / num_gt if num_gt > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum) if (tp_cumsum + fp_cumsum).sum() > 0 else np.zeros_like(tp_cumsum)
        
        # Calculate AP using 11-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
        
        ap_per_class[class_id] = ap
    
    return ap_per_class

def evaluate_model(model, data_loader, device, num_classes):
    """Evaluate the model and return metrics."""
    model.eval()
    
    all_predictions = []
    all_ground_truths = []
    
    logging.info("Running inference on test dataset...")
    print("Running inference on test dataset...")
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader):
            images = list(img.to(device) for img in images)
            
            # Get predictions
            predictions = model(images)
            
            # Filter predictions by confidence threshold
            filtered_predictions = []
            for pred in predictions:
                mask = pred['scores'] >= CONFIDENCE_THRESHOLD
                filtered_pred = {
                    'boxes': pred['boxes'][mask],
                    'labels': pred['labels'][mask],
                    'scores': pred['scores'][mask]
                }
                filtered_predictions.append(filtered_pred)
            
            all_predictions.extend(filtered_predictions)
            all_ground_truths.extend(targets)
            
            if (batch_idx + 1) % 10 == 0:
                progress_msg = f"Processed {batch_idx + 1}/{len(data_loader)} batches"
                logging.info(progress_msg)
                print(progress_msg)
    
    logging.info("Calculating metrics...")
    print("Calculating metrics...")
    # Calculate AP per class
    ap_per_class = calculate_ap_per_class(all_predictions, all_ground_truths, num_classes, IOU_THRESHOLD)
    
    # Calculate mAP (mean Average Precision)
    valid_aps = [ap for ap in ap_per_class.values() if ap > 0]
    mean_ap = np.mean(valid_aps) if len(valid_aps) > 0 else 0.0
    
    # Calculate overall statistics
    total_predictions = sum(len(pred['boxes']) for pred in all_predictions)
    total_ground_truths = sum(len(gt['boxes']) for gt in all_ground_truths)
    
    # Count predictions and ground truths per class
    pred_counts = defaultdict(int)
    gt_counts = defaultdict(int)
    
    for pred in all_predictions:
        for label in pred['labels'].cpu().numpy():
            pred_counts[int(label)] += 1
    
    for gt in all_ground_truths:
        for label in gt['labels'].cpu().numpy():
            gt_counts[int(label)] += 1
    
    # Compile evaluation results dictionary
    results = {
        'mean_average_precision': float(mean_ap),
        'ap_per_class': {str(k): float(v) for k, v in ap_per_class.items()},
        'total_predictions': int(total_predictions),
        'total_ground_truths': int(total_ground_truths),
        'predictions_per_class': {str(k): int(v) for k, v in pred_counts.items()},
        'ground_truths_per_class': {str(k): int(v) for k, v in gt_counts.items()},
        'num_images': len(all_predictions),
        'iou_threshold': IOU_THRESHOLD,
        'confidence_threshold': CONFIDENCE_THRESHOLD
    }
    
    return results

def get_model(num_classes):
    """
    Initialize Faster R-CNN model architecture with ResNet-50-FPN backbone.
    
    Args:
        num_classes: Number of object classes (including background)
    
    Returns:
        Initialized model with custom classifier head
    """
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found: {MODEL_PATH}")
        print("Please update MODEL_PATH in the script to point to your trained model.")
        return
    
    # Check if test list exists
    if not os.path.exists(TEST_LIST):
        print(f"ERROR: Test list file not found: {TEST_LIST}")
        print("Please run create_dataset.py first to generate train/test splits.")
        return
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    logging.info(f"--- STARTING MODEL EVALUATION ---")
    logging.info(f"Loading model from: {MODEL_PATH}")
    logging.info(f"Evaluating on TEST set: {TEST_LIST}")
    logging.info(f"Device: {DEVICE} | IoU Threshold: {IOU_THRESHOLD} | Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"Loading model from: {MODEL_PATH}")
    print(f"Evaluating on TEST set: {TEST_LIST}")
    print(f"Device: {DEVICE}")
    print("-" * 60)
    
    # Load model
    model = get_model(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    
    # Load test dataset
    dataset = DentalDataset(TEST_LIST, transforms=ToTensor())
    data_loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    logging.info(f"Test dataset size: {len(dataset)} images")
    print(f"Test dataset size: {len(dataset)} images")
    print("-" * 60)
    
    # Evaluate
    results = evaluate_model(model, data_loader, DEVICE, NUM_CLASSES)
    
    # Log and print results
    logging.info("=" * 60)
    logging.info("TEST SET EVALUATION RESULTS")
    logging.info("=" * 60)
    logging.info(f"Mean Average Precision (mAP@{int(IOU_THRESHOLD*100)}): {results['mean_average_precision']:.4f} ({results['mean_average_precision']*100:.2f}%)")
    logging.info(f"Total Images Evaluated: {results['num_images']}")
    logging.info(f"Total Predictions: {results['total_predictions']}")
    logging.info(f"Total Ground Truth Boxes: {results['total_ground_truths']}")
    logging.info("Average Precision per Class:")
    for class_id, ap in sorted(results['ap_per_class'].items(), key=lambda x: int(x[0])):
        if ap > 0:
            logging.info(f"  Class {class_id}: {ap:.4f} ({ap*100:.2f}%)")
    
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean Average Precision (mAP@{int(IOU_THRESHOLD*100)}): {results['mean_average_precision']:.4f} ({results['mean_average_precision']*100:.2f}%)")
    print(f"Total Images Evaluated: {results['num_images']}")
    print(f"Total Predictions: {results['total_predictions']}")
    print(f"Total Ground Truth Boxes: {results['total_ground_truths']}")
    print("\nAverage Precision per Class:")
    for class_id, ap in sorted(results['ap_per_class'].items(), key=lambda x: int(x[0])):
        if ap > 0:
            print(f"  Class {class_id}: {ap:.4f} ({ap*100:.2f}%)")
    
    # Save results to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to: {OUTPUT_FILE}")
    logging.info(f"Evaluation log saved to: {LOG_FILE}")
    logging.info(f"FINAL TEST ACCURACY (mAP): {results['mean_average_precision']*100:.2f}%")
    logging.info("--- EVALUATION COMPLETE ---")
    
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print(f"Evaluation log saved to: {LOG_FILE}")
    print("\n" + "=" * 60)
    print(f"FINAL TEST ACCURACY (mAP): {results['mean_average_precision']*100:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
