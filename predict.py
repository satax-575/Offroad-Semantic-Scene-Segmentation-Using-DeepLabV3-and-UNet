"""
Ensemble Inference: UNet + DeepLabV3 + TTA
"""

import torch
from test import load_ensemble_models, ensemble_predict
from config import NUM_CLASSES, DEVICE
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path

# FIXED: Use Path to handle Windows paths correctly
UNET_PATH = str(Path("checkpoints/best_unet_model.pth"))
DEEPLAB_PATH = str(Path("checkpoints/best_deeplab_model.pth"))

def load_image(image_path):
    """Load and preprocess image"""
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img)

def predict(image_path, output_path=None):
    """Predict using ensemble of both models with TTA"""
    print("Loading models...")
    models = load_ensemble_models([UNET_PATH, DEEPLAB_PATH], "unet", NUM_CLASSES, DEVICE)
    
    print(f"Predicting: {image_path}")
    image = load_image(image_path).to(DEVICE)
    
    prediction = ensemble_predict(models, image, use_tta=True)
    prediction = torch.argmax(prediction, dim=1).squeeze(0).cpu().numpy()
    
    if output_path:
        Image.fromarray(prediction.astype(np.uint8)).save(output_path)
        print(f"Saved: {output_path}")
    
    return prediction

if __name__ == "__main__":
    print("Ensemble ready! Use: predict('test.png', 'output.png')")
