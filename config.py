import torch
from pathlib import Path

# Device configuration with detailed GPU info
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"CUDA version: {torch.version.cuda}")
else:
    DEVICE = torch.device("cpu")
    print("No GPU detected, using CPU")
    print("Note: Training will be significantly slower on CPU")

DATA_DIR = "./data/Offroad_Segmentation_testImages/Offroad_Segmentation_testImages"
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Class mapping (EDIT if needed)
CLASS_IDS = [0, 100, 200, 300, 7100, 10000]
CLASS_ID_TO_INDEX = {cid: i for i, cid in enumerate(CLASS_IDS)}
INDEX_TO_CLASS_ID = {i: cid for i, cid in enumerate(CLASS_IDS)}

NUM_CLASSES = len(CLASS_IDS)

TRAIN_CONFIG = {
    "img_height": 256,
    "img_width": 256,

    "batch_size": 2,
    "grad_accum": 4,

    "epochs": 60,  # IMPROVED: Increased from 40 to 60 for better convergence
    "lr": 3e-4,  # Stable learning rate

    "mixed_precision": True,
    "loss": "competition",

    "patch_size": 320,  # Larger context (fallback to 256 if OOM)
    
    # IMPROVED: Model selection
    "model_type": "deeplab",  # Training DeepLabV3
}
