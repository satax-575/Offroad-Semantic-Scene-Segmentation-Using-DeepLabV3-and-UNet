"""
Automated script to train both UNet and DeepLabV3, then ensemble them
Run this once and it will handle everything!
"""

import torch
import subprocess
import sys
from pathlib import Path
import json
import time

def update_config(model_type):
    """Update config.py to use specified model type"""
    print(f"\n{'='*60}")
    print(f"Updating config.py to use {model_type.upper()}")
    print(f"{'='*60}\n")
    
    # Read current config
    with open('config.py', 'r') as f:
        lines = f.readlines()
    
    # Update model_type line
    new_lines = []
    for line in lines:
        if '"model_type"' in line and ':' in line:
            # Replace the model type
            indent = line[:line.index('"')]
            new_lines.append(f'{indent}"model_type": "{model_type}",  # Auto-updated by train_both_models.py\n')
        else:
            new_lines.append(line)
    
    # Write updated config
    with open('config.py', 'w') as f:
        f.writelines(new_lines)
    
    print(f"✓ Config updated to use {model_type}")


def train_model(model_name):
    """Train a single model"""
    print(f"\n{'='*60}")
    print(f"TRAINING {model_name.upper()} MODEL")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Run training
    result = subprocess.run([sys.executable, 'train.py'], 
                          capture_output=False, 
                          text=True)
    
    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    
    if result.returncode == 0:
        print(f"\n✓ {model_name.upper()} training completed successfully!")
        print(f"  Time taken: {hours}h {minutes}m")
        return True
    else:
        print(f"\n✗ {model_name.upper()} training failed!")
        return False


def find_best_model(model_type):
    """Find the best model checkpoint for given type"""
    checkpoint_dir = Path("checkpoints")
    
    # Look for best model files
    best_models = list(checkpoint_dir.glob("best_model_*.pth"))
    
    if not best_models:
        print(f"Warning: No best model found for {model_type}")
        return None
    
    # Get the most recent one (highest IoU)
    best_models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    best_model = best_models[0]
    
    print(f"Found best {model_type} model: {best_model.name}")
    return str(best_model)


def rename_model(old_path, model_type):
    """Rename model to include type"""
    if not old_path:
        return None
    
    old_path = Path(old_path)
    new_name = f"best_{model_type}_model.pth"
    new_path = old_path.parent / new_name
    
    # Copy instead of rename to preserve original
    import shutil
    shutil.copy(old_path, new_path)
    
    print(f"✓ Saved as: {new_path}")
    return str(new_path)


def create_ensemble_script(unet_path, deeplab_path):
    """Create a script to use the ensemble"""
    script = f'''"""
Ensemble Inference Script
Automatically generated after training both models
"""

import torch
from test import load_ensemble_models, ensemble_predict
from config import NUM_CLASSES, DEVICE
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

# Model paths
UNET_PATH = "{unet_path}"
DEEPLAB_PATH = "{deeplab_path}"

def load_image(image_path):
    """Load and preprocess image"""
    img = Image.open(image_path).convert("RGB")
    
    # Same preprocessing as training
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return transform(img)


def predict_single_image(image_path, output_path=None):
    """
    Predict using ensemble of both models with TTA
    
    Args:
        image_path: path to input image
        output_path: path to save prediction (optional)
    
    Returns:
        prediction array (H, W)
    """
    print(f"Loading models...")
    
    # Load both models
    models = load_ensemble_models(
        [UNET_PATH, DEEPLAB_PATH],
        "unet",  # Will auto-detect correct type
        NUM_CLASSES,
        DEVICE
    )
    
    print(f"Predicting: {{image_path}}")
    
    # Load image
    image = load_image(image_path).to(DEVICE)
    
    # Predict with ensemble + TTA
    prediction = ensemble_predict(models, image, use_tta=True)
    prediction = torch.argmax(prediction, dim=1).squeeze(0)
    
    # Convert to numpy
    prediction = prediction.cpu().numpy()
    
    if output_path:
        # Save prediction
        from PIL import Image
        pred_img = Image.fromarray(prediction.astype(np.uint8))
        pred_img.save(output_path)
        print(f"Saved prediction to: {{output_path}}")
    
    return prediction


def predict_batch(image_folder, output_folder):
    """
    Predict on all images in a folder
    
    Args:
        image_folder: folder containing input images
        output_folder: folder to save predictions
    """
    from pathlib import Path
    
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    
    # Get all images
    image_files = list(image_folder.glob("*.png")) + \\
                  list(image_folder.glob("*.jpg")) + \\
                  list(image_folder.glob("*.jpeg"))
    
    print(f"Found {{len(image_files)}} images")
    
    for img_path in image_files:
        output_path = output_folder / f"{{img_path.stem}}_pred.png"
        predict_single_image(str(img_path), str(output_path))
    
    print(f"\\nAll predictions saved to: {{output_folder}}")


if __name__ == "__main__":
    print("="*60)
    print("ENSEMBLE PREDICTION (UNet + DeepLabV3 + TTA)")
    print("="*60)
    
    # Example usage
    print("\\nExample 1: Single image prediction")
    print("  prediction = predict_single_image('test.png', 'output.png')")
    
    print("\\nExample 2: Batch prediction")
    print("  predict_batch('test_images/', 'predictions/')")
    
    print("\\n" + "="*60)
    
    # Uncomment to test:
    # predict_single_image("path/to/test/image.png", "prediction.png")
'''
    
    with open('ensemble_predict.py', 'w') as f:
        f.write(script)
    
    print(f"\n✓ Created ensemble_predict.py")
    print(f"  Use this script for inference with both models!")


def main():
    print("="*60)
    print("AUTOMATED TRAINING: UNet + DeepLabV3 + Ensemble")
    print("="*60)
    print("\nThis will:")
    print("1. Train UNet model (60 epochs, ~2-3 hours)")
    print("2. Train DeepLabV3 model (60 epochs, ~4-5 hours)")
    print("3. Create ensemble inference script")
    print("\nTotal time: ~6-8 hours")
    print("="*60)
    
    input("\nPress Enter to start training (or Ctrl+C to cancel)...")
    
    results = {}
    
    # Train UNet
    print("\n" + "="*60)
    print("PHASE 1/2: Training UNet")
    print("="*60)
    
    update_config("unet")
    if train_model("unet"):
        unet_path = find_best_model("unet")
        unet_path = rename_model(unet_path, "unet")
        results['unet'] = unet_path
    else:
        print("UNet training failed. Stopping.")
        return
    
    # Train DeepLabV3
    print("\n" + "="*60)
    print("PHASE 2/2: Training DeepLabV3")
    print("="*60)
    
    update_config("deeplab")
    if train_model("deeplab"):
        deeplab_path = find_best_model("deeplab")
        deeplab_path = rename_model(deeplab_path, "deeplab")
        results['deeplab'] = deeplab_path
    else:
        print("DeepLabV3 training failed.")
        return
    
    # Create ensemble script
    print("\n" + "="*60)
    print("CREATING ENSEMBLE INFERENCE SCRIPT")
    print("="*60)
    
    create_ensemble_script(results['unet'], results['deeplab'])
    
    # Save results
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final summary
    print("\n" + "="*60)
    print("✓ ALL TRAINING COMPLETE!")
    print("="*60)
    print(f"\nTrained models:")
    print(f"  UNet:      {results['unet']}")
    print(f"  DeepLabV3: {results['deeplab']}")
    print(f"\nInference script: ensemble_predict.py")
    print(f"\nTo use ensemble:")
    print(f"  python ensemble_predict.py")
    print(f"  # Then call predict_single_image() or predict_batch()")
    print("\n" + "="*60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining cancelled by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
