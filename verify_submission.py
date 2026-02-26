"""
Submission Verification Script
Run this to verify all files are present and working
"""

import os
from pathlib import Path
import sys

def check_file(filepath, description):
    """Check if a file exists"""
    if Path(filepath).exists():
        size = Path(filepath).stat().st_size
        size_mb = size / (1024 * 1024)
        print(f"✓ {description:40s} ({size_mb:.2f} MB)")
        return True
    else:
        print(f"✗ {description:40s} MISSING!")
        return False

def check_import(module_name):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✓ {module_name:40s}")
        return True
    except ImportError:
        print(f"✗ {module_name:40s} NOT INSTALLED!")
        return False

def main():
    print("="*60)
    print("COMPETITION SUBMISSION VERIFICATION")
    print("="*60)
    
    all_good = True
    
    # Check Python version
    print("\n1. Python Version:")
    print(f"   {sys.version}")
    if sys.version_info < (3, 9):
        print("   ⚠️  Warning: Python 3.9+ recommended")
    
    # Check core files
    print("\n2. Core Python Files:")
    files = [
        ("config.py", "Configuration"),
        ("model.py", "Model architectures"),
        ("dataset.py", "Data loading"),
        ("utils.py", "Loss functions"),
        ("train.py", "Training script"),
        ("test.py", "Inference tools"),
        ("predict.py", "Single prediction"),
        ("batch_predict.py", "Batch prediction"),
        ("evaluate.py", "Evaluation metrics"),
        ("train_both_models.py", "Automated training"),
        ("app.py", "Web interface"),
    ]
    
    for filename, desc in files:
        if not check_file(filename, desc):
            all_good = False
    
    # Check documentation
    print("\n3. Documentation Files:")
    docs = [
        ("README.md", "Main documentation"),
        ("PROJECT_REPORT.md", "Technical report"),
        ("SUBMISSION_GUIDE.md", "Submission guide"),
        ("GITHUB_SETUP.md", "GitHub setup"),
        ("requirements.txt", "Dependencies"),
    ]
    
    for filename, desc in docs:
        if not check_file(filename, desc):
            all_good = False
    
    # Check models
    print("\n4. Pre-trained Models:")
    models = [
        ("checkpoints/best_unet_model.pth", "UNet model"),
        ("checkpoints/best_deeplab_model.pth", "DeepLabV3 model"),
    ]
    
    for filename, desc in models:
        if not check_file(filename, desc):
            all_good = False
    
    # Check templates
    print("\n5. Web Interface:")
    if not check_file("templates/index.html", "HTML template"):
        all_good = False
    
    # Check dependencies
    print("\n6. Python Dependencies:")
    deps = [
        "torch",
        "torchvision",
        "numpy",
        "cv2",
        "PIL",
        "tqdm",
        "flask",
    ]
    
    for dep in deps:
        if not check_import(dep):
            all_good = False
    
    # Check GPU
    print("\n7. GPU Availability:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("⚠️  No GPU detected (CPU mode)")
            print("   Training will be very slow on CPU")
    except:
        print("✗ Cannot check GPU (torch not installed)")
    
    # Test imports
    print("\n8. Testing Core Imports:")
    try:
        from config import NUM_CLASSES, DEVICE
        print(f"✓ config.py imports successfully")
        print(f"  NUM_CLASSES: {NUM_CLASSES}")
        print(f"  DEVICE: {DEVICE}")
    except Exception as e:
        print(f"✗ config.py import failed: {e}")
        all_good = False
    
    try:
        from model import UNet, DeepLabV3Wrapper, get_model
        print(f"✓ model.py imports successfully")
    except Exception as e:
        print(f"✗ model.py import failed: {e}")
        all_good = False
    
    try:
        from utils import BetterLoss, iou_score
        print(f"✓ utils.py imports successfully")
    except Exception as e:
        print(f"✗ utils.py import failed: {e}")
        all_good = False
    
    # Summary
    print("\n" + "="*60)
    if all_good:
        print("✅ ALL CHECKS PASSED!")
        print("="*60)
        print("\nYour submission is ready!")
        print("\nNext steps:")
        print("1. Upload to GitHub (see GITHUB_SETUP.md)")
        print("2. Test with: python batch_predict.py test_images/ predictions/")
        print("3. Submit GitHub URL to competition")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("="*60)
        print("\nPlease fix the issues above before submitting.")
        print("\nCommon fixes:")
        print("- Install missing dependencies: pip install -r requirements.txt")
        print("- Ensure all files are in the correct location")
        print("- Check that model files are not corrupted")
    
    print("\n" + "="*60)
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
