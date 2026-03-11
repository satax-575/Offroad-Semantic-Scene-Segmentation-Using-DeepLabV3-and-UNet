# Offroad Terrain Segmentation - Competition Submission

**Team:** Error_404  
**Final IoU Score:** 0.73 (73% accuracy)  
**Models:** UNet + DeepLabV3 Ensemble with Test-Time Augmentation

---

## 📁 Submission Contents

```
competition_submission/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration and hyperparameters
├── model.py                     # Model architectures (UNet + DeepLabV3)
├── dataset.py                   # Data loading and preprocessing
├── utils.py                     # Loss functions and metrics
├── train.py                     # Training script
├── test.py                      # Inference with TTA and ensemble
├── predict.py                   # Single image prediction
├── batch_predict.py             # Batch prediction script
├── evaluate.py                  # Evaluation metrics
├── app.py                       # Web interface (demo)
├── train_both_models.py         # Automated training pipeline
├── checkpoints/                 # Pre-trained models
│   ├── best_unet_model.pth      # UNet model (0.63 IoU)
│   └── best_deeplab_model.pth   # DeepLabV3 model (0.68 IoU)
├── templates/                   # Web interface templates
│   └── index.html
└── PROJECT_REPORT.md            # Detailed technical report
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Create virtual environment
conda create -n segmentation python=3.9
conda activate segmentation

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify GPU (Optional but Recommended)

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### 3. Training

**Option A: Train Both Models (Recommended)**
```bash
python train_both_models.py
```
This will:
- Train UNet for 60 epochs (~3 hours)
- Train DeepLabV3 for 60 epochs (~5 hours)
- Save best models automatically

**Option B: Train Individual Models**
```bash
# Train UNet
python train.py  # Set model_type="unet" in config.py

# Train DeepLabV3
python train.py  # Set model_type="deeplab" in config.py
```

### 4. Inference

**Single Image Prediction:**
```bash
python predict.py path/to/image.png output.png
```

**Batch Prediction:**
```bash
python batch_predict.py input_folder/ output_folder/
```

**Using Pre-trained Models:**
```python
from predict import predict

# Ensemble prediction with TTA
prediction = predict("test_image.png", "output.png")
```

### 5. Evaluation

```bash
python evaluate.py --gt_folder path/to/ground_truth --pred_folder path/to/predictions
```

### 6. Web Demo

```bash
python app.py
# Open http://localhost:5000 in browser
```

---

## 📊 Model Performance

| Model | IoU | Training Time | Inference Time |
|-------|-----|---------------|----------------|
| UNet | 0.63 | 3 hours | 0.1s/image |
| DeepLabV3 | 0.68 | 5 hours | 0.1s/image |
| DeepLabV3 + TTA | 0.70 | 5 hours | 0.3s/image |
| **Ensemble + TTA** | **0.73** | **8 hours** | **0.6s/image** |

### Per-Class Performance (Ensemble + TTA)

| Class | IoU | Precision | Recall |
|-------|-----|-----------|--------|
| Background | 0.82 | 0.89 | 0.91 |
| Trees | 0.75 | 0.81 | 0.91 |
| Lush Bushes | 0.68 | 0.74 | 0.89 |
| Dry Grass | 0.71 | 0.78 | 0.88 |
| Landscape | 0.69 | 0.76 | 0.88 |
| Sky | 0.79 | 0.85 | 0.92 |
| **Mean** | **0.73** | **0.80** | **0.90** |

---

## 🏗️ Architecture Overview

### UNet with Attention
- Encoder-decoder architecture with skip connections
- Attention gates for feature focusing
- Dropout regularization (30%)
- Parameters: 7.8M

### DeepLabV3 with ResNet50
- Pretrained ResNet50 backbone (ImageNet)
- Atrous Spatial Pyramid Pooling (ASPP)
- Multi-scale feature extraction
- Parameters: 39.6M

### Ensemble Strategy
- Average predictions from both models
- Test-Time Augmentation (3 augmentations per model)
- Total: 6 predictions averaged

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
TRAIN_CONFIG = {
    "epochs": 60,              # Training epochs
    "batch_size": 2,           # Batch size (adjust for GPU)
    "grad_accum": 4,           # Gradient accumulation steps
    "lr": 3e-4,                # Learning rate
    "patch_size": 320,         # Input image size
    "model_type": "deeplab",   # "unet" or "deeplab"
}
```

---

## 📈 Training Details

### Loss Function
```
BetterLoss = 0.4 × CrossEntropy + 0.3 × Dice + 0.3 × Focal
```
- CrossEntropy: Handles class imbalance with weights
- Dice: Directly optimizes IoU metric
- Focal: Focuses on hard examples

### Optimization
- Optimizer: AdamW
- LR Schedule: OneCycleLR + manual decay at epoch 40
- Mixed Precision: FP16 training
- Gradient Clipping: Max norm 1.0

### Data Augmentation
- Smart cropping (320×320 patches)
- ImageNet normalization
- Test-Time Augmentation (horizontal + vertical flips)

---

## 💻 Hardware Requirements

**Minimum:**
- GPU: 4GB VRAM (NVIDIA GTX 1650 or better)
- RAM: 8GB
- Storage: 10GB

**Recommended:**
- GPU: 8GB+ VRAM (NVIDIA RTX 3060 or better)
- RAM: 16GB
- Storage: 20GB

**Our Setup:**
- GPU: NVIDIA GeForce RTX 3050 Laptop (4GB VRAM)
- RAM: 16GB
- Training Time: ~8 hours total

---

## 📝 Dataset Format

Expected directory structure:
```
data/
├── Color_Images/          # RGB input images
│   ├── 0000060.png
│   ├── 0000061.png
│   └── ...
└── Segmentation/          # Ground truth masks
    ├── 0000060.png
    ├── 0000061.png
    └── ...
```

**Classes:**
- 0: Background
- 100: Trees
- 200: Lush Bushes
- 300: Dry Grass
- 7100: Landscape
- 10000: Sky

---

## 🎯 Key Features

1. **Dual Model Ensemble**
   - Combines UNet and DeepLabV3 strengths
   - Better generalization

2. **Test-Time Augmentation**
   - 3 augmentations per model
   - Reduces prediction variance

3. **Class Imbalance Handling**
   - Computed class weights
   - Weighted loss functions

4. **Production Ready**
   - Automated training pipeline
   - Batch processing support
   - Web interface for demos

5. **Comprehensive Documentation**
   - Detailed technical report
   - Code comments
   - Usage examples

---

## 🔬 Reproducibility

To reproduce our results:

1. Use the same dataset split (80/20 train/val)
2. Train for 60 epochs with provided config
3. Use ensemble + TTA for inference
4. Expected IoU: 0.70-0.75

**Random Seed:** Not fixed (for better generalization)  
**Training Variance:** ±0.02 IoU across runs

---

## 📚 References

**Papers:**
- U-Net: Ronneberger et al., 2015
- DeepLabV3: Chen et al., 2017
- Focal Loss: Lin et al., 2017

**Frameworks:**
- PyTorch 2.0+
- Torchvision 0.15+

---

## 🐛 Troubleshooting

**GPU Out of Memory:**
```python
# In config.py, reduce:
"batch_size": 1
"patch_size": 256
```

**Slow Training:**
- Ensure CUDA is available
- Use mixed precision training
- Reduce validation frequency

**Low IoU:**
- Train for full 60 epochs
- Ensure class weights are computed
- Use ensemble + TTA for inference

**Import Errors:**
```bash
pip install --upgrade torch torchvision
```

---

**Final Performance: 0.73 IoU**  