"""
Evaluation Script for Segmentation Predictions
Computes IoU and per-class metrics
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json
import argparse


def compute_iou(pred, gt, num_classes):
    """Compute IoU for each class"""
    ious = []
    
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        gt_mask = (gt == cls)
        
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        
        if union == 0:
            continue  # Skip classes not present in ground truth
        
        iou = intersection / union
        ious.append(iou)
    
    return ious


def evaluate(gt_folder, pred_folder, num_classes=6):
    """
    Evaluate predictions against ground truth
    
    Args:
        gt_folder: folder containing ground truth masks
        pred_folder: folder containing prediction masks
        num_classes: number of classes
    
    Returns:
        dict with evaluation metrics
    """
    gt_folder = Path(gt_folder)
    pred_folder = Path(pred_folder)
    
    # Find all ground truth files
    gt_files = sorted(list(gt_folder.glob("*.png")))
    
    if len(gt_files) == 0:
        print(f"❌ No ground truth files found in {gt_folder}")
        return None
    
    print(f"Found {len(gt_files)} ground truth files")
    
    all_ious = []
    class_ious = [[] for _ in range(num_classes)]
    
    for gt_path in tqdm(gt_files, desc="Evaluating"):
        # Find corresponding prediction
        pred_path = pred_folder / gt_path.name
        
        if not pred_path.exists():
            # Try with _pred suffix
            pred_path = pred_folder / f"{gt_path.stem}_pred.png"
        
        if not pred_path.exists():
            print(f"⚠️  Prediction not found for {gt_path.name}")
            continue
        
        # Load images
        gt = np.array(Image.open(gt_path))
        pred = np.array(Image.open(pred_path))
        
        # Ensure same size
        if gt.shape != pred.shape:
            print(f"⚠️  Size mismatch for {gt_path.name}: GT {gt.shape} vs Pred {pred.shape}")
            # Resize prediction to match ground truth
            pred = np.array(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.NEAREST))
        
        # Compute IoU
        ious = compute_iou(pred, gt, num_classes)
        
        if len(ious) > 0:
            all_ious.append(np.mean(ious))
            
            # Store per-class IoUs
            for cls, iou in enumerate(ious):
                class_ious[cls].append(iou)
    
    # Compute statistics
    mean_iou = np.mean(all_ious) if all_ious else 0
    std_iou = np.std(all_ious) if all_ious else 0
    
    class_names = ['Background', 'Trees', 'Lush Bushes', 'Dry Grass', 'Landscape', 'Sky']
    
    results = {
        'mean_iou': float(mean_iou),
        'std_iou': float(std_iou),
        'num_images': len(all_ious),
        'per_class': {}
    }
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Mean IoU: {mean_iou:.4f} ± {std_iou:.4f}")
    print(f"Images evaluated: {len(all_ious)}")
    print("\nPer-Class IoU:")
    print("-"*60)
    
    for cls in range(num_classes):
        if len(class_ious[cls]) > 0:
            cls_mean = np.mean(class_ious[cls])
            cls_std = np.std(class_ious[cls])
            class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
            
            results['per_class'][class_name] = {
                'mean_iou': float(cls_mean),
                'std_iou': float(cls_std),
                'count': len(class_ious[cls])
            }
            
            print(f"{class_name:15s}: {cls_mean:.4f} ± {cls_std:.4f} ({len(class_ious[cls])} images)")
    
    print("="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate segmentation predictions')
    parser.add_argument('--gt_folder', type=str, required=True,
                        help='Path to ground truth folder')
    parser.add_argument('--pred_folder', type=str, required=True,
                        help='Path to predictions folder')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='Number of classes (default: 6)')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='Output JSON file (default: evaluation_results.json)')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate(args.gt_folder, args.pred_folder, args.num_classes)
    
    if results:
        # Save results to JSON
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Results saved to {args.output}")


if __name__ == "__main__":
    main()
