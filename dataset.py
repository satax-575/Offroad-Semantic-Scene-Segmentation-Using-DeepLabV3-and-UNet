import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from config import CLASS_ID_TO_INDEX, TRAIN_CONFIG


def smart_crop(img, mask, size):
    h, w = mask.shape
    for _ in range(10):
        y = np.random.randint(0, h - size)
        x = np.random.randint(0, w - size)
        m = mask[y:y+size, x:x+size]
        if len(np.unique(m)) > 1:
            return img[y:y+size, x:x+size], m
    return img[:size, :size], mask[:size, :size]


class SegDataset(Dataset):
    def __init__(self, root, split="train"):
        # For the test images dataset, there's no train/val split
        # The structure is: root/Color_Images and root/Segmentation
        if split is None or split == "none":
            # No split - use root directly
            self.img_dir = Path(root)/"Color_Images"
            self.mask_dir = Path(root)/"Segmentation"
        elif split in ["train", "val"]:
            # Check if split directories exist
            split_path = Path(root)/split/"Color_Images"
            if split_path.exists():
                # If you have a training dataset with splits, use this
                self.img_dir = Path(root)/split/"Color_Images"
                self.mask_dir = Path(root)/split/"Segmentation"
            else:
                # Fallback to root if split doesn't exist
                self.img_dir = Path(root)/"Color_Images"
                self.mask_dir = Path(root)/"Segmentation"
        else:
            # For test images without splits
            self.img_dir = Path(root)/"Color_Images"
            self.mask_dir = Path(root)/"Segmentation"

        self.files = sorted(list(self.img_dir.glob("*.png")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        mask_path = self.mask_dir / img_path.name

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        # map labels
        mapped = np.zeros_like(mask)
        for k, v in CLASS_ID_TO_INDEX.items():
            mapped[mask == k] = v

        img, mapped = smart_crop(img, mapped, TRAIN_CONFIG["patch_size"])

        # FIXED: Add proper normalization with ImageNet stats
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # FIXED: Ensure float32 tensor (not float64)
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)

        mask = torch.tensor(mapped).long()

        return img, mask


def get_loader(data_dir):
    # Check if train/val splits exist
    train_path = Path(data_dir) / "train" / "Color_Images"
    
    if train_path.exists():
        # Use train/val splits if they exist
        train_ds = SegDataset(data_dir, "train")
        val_ds = SegDataset(data_dir, "val")
    else:
        # No splits - use the same dataset for both train and val
        # Split 80/20 manually
        full_ds = SegDataset(data_dir, split=None)
        train_size = int(0.8 * len(full_ds))
        val_size = len(full_ds) - train_size
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
        print(f"No train/val split found. Created split: {train_size} train, {val_size} val")

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    return train_loader, val_loader