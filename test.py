import torch
from model import get_model
from config import *


def tta_predict(model, img):
    """
    IMPROVED: Test Time Augmentation with horizontal and vertical flips
    
    Args:
        model: trained model
        img: input image tensor (C, H, W)
    
    Returns:
        averaged prediction tensor (1, C, H, W)
    """
    preds = []

    # Original + horizontal flip + vertical flip
    # For (C, H, W) tensor: dim 1 is H (vertical), dim 2 is W (horizontal)
    for flip in [None, 1, 2]:  # FIXED: Changed from [None, 2, 3] to [None, 1, 2]
        x = img.clone()

        if flip == 1:  # Vertical flip (flip height dimension)
            x = torch.flip(x, [1])
        elif flip == 2:  # Horizontal flip (flip width dimension)
            x = torch.flip(x, [2])

        out = model(x.unsqueeze(0))
        p = torch.softmax(out, dim=1)

        # Flip back
        if flip == 1:
            p = torch.flip(p, [2])  # Flip back height (now dim 2 in batch)
        elif flip == 2:
            p = torch.flip(p, [3])  # Flip back width (now dim 3 in batch)

        preds.append(p)

    return torch.mean(torch.stack(preds), dim=0)


def sliding_window_predict(model, img, size=256, stride=128):
    """
    IMPROVED: Sliding window inference for large images
    
    Args:
        model: trained model
        img: input image tensor (C, H, W)
        size: window size
        stride: stride for sliding
    
    Returns:
        prediction tensor (1, num_classes, H, W)
    """
    C, H, W = img.shape
    num_classes = model.out.out_channels if hasattr(model, 'out') else 6
    
    output = torch.zeros(1, num_classes, H, W).to(img.device)
    count = torch.zeros(1, 1, H, W).to(img.device)
    
    for y in range(0, H - size + 1, stride):
        for x in range(0, W - size + 1, stride):
            patch = img[:, y:y+size, x:x+size]
            pred = torch.softmax(model(patch.unsqueeze(0)), dim=1)
            output[:, :, y:y+size, x:x+size] += pred
            count[:, :, y:y+size, x:x+size] += 1
    
    # Handle edges if image size not divisible by stride
    if H % stride != 0:
        y = H - size
        for x in range(0, W - size + 1, stride):
            patch = img[:, y:y+size, x:x+size]
            pred = torch.softmax(model(patch.unsqueeze(0)), dim=1)
            output[:, :, y:y+size, x:x+size] += pred
            count[:, :, y:y+size, x:x+size] += 1
    
    if W % stride != 0:
        x = W - size
        for y in range(0, H - size + 1, stride):
            patch = img[:, y:y+size, x:x+size]
            pred = torch.softmax(model(patch.unsqueeze(0)), dim=1)
            output[:, :, y:y+size, x:x+size] += pred
            count[:, :, y:y+size, x:x+size] += 1
    
    return output / (count + 1e-6)


def ensemble_predict(models, img, use_tta=True):
    """
    IMPROVED: Ensemble prediction from multiple models
    
    Args:
        models: list of trained models
        img: input image tensor (C, H, W)
        use_tta: whether to use TTA for each model
    
    Returns:
        averaged prediction tensor (1, C, H, W)
    """
    preds = []
    
    for model in models:
        model.eval()
        if use_tta:
            p = tta_predict(model, img)
        else:
            with torch.no_grad():
                p = torch.softmax(model(img.unsqueeze(0)), dim=1)
        preds.append(p)
    
    return torch.mean(torch.stack(preds), dim=0)


def predict_with_tta(model, img, device):
    """
    Predict with Test Time Augmentation
    
    Args:
        model: trained model
        img: input image tensor (C, H, W)
        device: torch device
    
    Returns:
        prediction tensor (H, W)
    """
    model.eval()
    img = img.to(device)
    
    with torch.no_grad():
        pred = tta_predict(model, img)
        pred = torch.argmax(pred, dim=1).squeeze(0)
    
    return pred


def load_ensemble_models(checkpoint_paths, model_type, num_classes, device):
    """
    Load multiple models for ensemble
    
    Args:
        checkpoint_paths: list of paths to model checkpoints
        model_type: "unet" or "deeplab" (can be mixed)
        num_classes: number of classes
        device: torch device
    
    Returns:
        list of loaded models
    """
    models = []
    
    for path in checkpoint_paths:
        # Auto-detect model type from checkpoint
        checkpoint = torch.load(path, map_location=device)
        
        # Check if it's a DeepLabV3 model (has 'model.backbone' keys)
        is_deeplab = any('model.backbone' in key for key in checkpoint.keys())
        
        if is_deeplab:
            print(f"Loading DeepLabV3 from: {path}")
            model = get_model("deeplab", num_classes).to(device)
        else:
            print(f"Loading UNet from: {path}")
            model = get_model("unet", num_classes).to(device)
        
        model.load_state_dict(checkpoint)
        model.eval()
        models.append(model)
    
    return models


if __name__ == "__main__":
    print("=" * 60)
    print("COMPETITION-LEVEL INFERENCE TOOLS")
    print("=" * 60)
    print("\nAvailable functions:")
    print("1. tta_predict(model, img) - Test Time Augmentation")
    print("2. sliding_window_predict(model, img) - For large images")
    print("3. ensemble_predict(models, img) - Ensemble multiple models")
    print("4. predict_with_tta(model, img, device) - Simple TTA prediction")
    print("\nExample usage:")
    print("  from test import predict_with_tta, ensemble_predict")
    print("  prediction = predict_with_tta(model, image, device)")
    print("=" * 60)
