"""
Batch Prediction Script
Predict on all test images using ensemble (UNet + DeepLabV3 + TTA)
"""

import torch
from predict import predict
from pathlib import Path
from tqdm import tqdm
import time

def batch_predict(input_folder, output_folder):
    """
    Predict on all images in a folder
    
    Args:
        input_folder: folder containing test images
        output_folder: folder to save predictions
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    
    # Find all images
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_folder.glob(ext))
    
    image_files = sorted(image_files)
    
    print("="*60)
    print("BATCH PREDICTION WITH ENSEMBLE")
    print("="*60)
    print(f"Input folder:  {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Found {len(image_files)} images")
    print("="*60)
    
    if len(image_files) == 0:
        print("\n❌ No images found!")
        print(f"Looking for images in: {input_folder}")
        print(f"Supported formats: {image_extensions}")
        return
    
    # Process each image
    start_time = time.time()
    
    for i, img_path in enumerate(tqdm(image_files, desc="Predicting")):
        output_path = output_folder / f"{img_path.stem}_pred.png"
        
        try:
            prediction = predict(str(img_path), str(output_path))
            
            # Show progress every 10 images
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(image_files) - i - 1)
                print(f"\nProcessed {i+1}/{len(image_files)} | "
                      f"Avg: {avg_time:.2f}s/img | "
                      f"ETA: {remaining/60:.1f}min")
        
        except Exception as e:
            print(f"\n❌ Error processing {img_path.name}: {e}")
            continue
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "="*60)
    print("✓ BATCH PREDICTION COMPLETE!")
    print("="*60)
    print(f"Processed: {len(image_files)} images")
    print(f"Time taken: {elapsed/60:.1f} minutes")
    print(f"Average: {elapsed/len(image_files):.2f} seconds per image")
    print(f"Predictions saved to: {output_folder}")
    print("="*60)


def predict_test_dataset():
    """Predict on the test dataset"""
    # Default test dataset path
    test_images = "data/Offroad_Segmentation_testImages/Offroad_Segmentation_testImages/Color_Images"
    output_folder = "predictions_ensemble"
    
    batch_predict(test_images, output_folder)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 3:
        # Custom input/output folders
        input_folder = sys.argv[1]
        output_folder = sys.argv[2]
        batch_predict(input_folder, output_folder)
    else:
        # Use default test dataset
        print("Usage: python batch_predict.py [input_folder] [output_folder]")
        print("Or just run without arguments to use default test dataset\n")
        
        predict_test_dataset()
