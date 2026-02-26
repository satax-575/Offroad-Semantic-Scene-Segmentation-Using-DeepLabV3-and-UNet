import torch
from torch.amp import autocast, GradScaler
import torch.optim as optim
from tqdm import tqdm

from config import *
from dataset import get_loader
from model import get_model
from utils import BetterLoss, iou_score


def compute_class_weights(loader, device):
    """Compute class weights for imbalanced dataset"""
    print("\nComputing class weights...")
    counts = torch.zeros(NUM_CLASSES)
    
    for img, mask in tqdm(loader, desc="Computing weights"):
        for c in range(NUM_CLASSES):
            counts[c] += (mask == c).sum()
    
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * NUM_CLASSES
    
    print(f"Class counts: {counts.tolist()}")
    print(f"Class weights: {weights.tolist()}")
    
    return weights.to(device)


def train():
    print("=" * 60)
    print("COMPETITION-LEVEL TRAINING PIPELINE")
    print("=" * 60)
    
    train_loader, val_loader = get_loader(DATA_DIR)

    # Print device info
    print(f"\nUsing device: {DEVICE}")
    if DEVICE.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.backends.cudnn.benchmark = True
    
    print(f"\nDataset: {DATA_DIR}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Epochs: {TRAIN_CONFIG['epochs']}")
    print(f"Learning rate: {TRAIN_CONFIG['lr']}")
    print(f"Patch size: {TRAIN_CONFIG['patch_size']}")
    print(f"Model type: {TRAIN_CONFIG['model_type']}")
    print("=" * 60 + "\n")

    # IMPROVED: Use model factory
    model = get_model(TRAIN_CONFIG['model_type'], NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=TRAIN_CONFIG["lr"])

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=TRAIN_CONFIG["lr"],
        steps_per_epoch=len(train_loader),
        epochs=TRAIN_CONFIG["epochs"]
    )

    # Compute class weights and use BetterLoss
    try:
        weights = compute_class_weights(train_loader, DEVICE)
        criterion = BetterLoss(weight=weights)
        print("Using BetterLoss with class weights\n")
    except Exception as e:
        print(f"Warning: Could not compute class weights ({e}), using uniform weights")
        criterion = BetterLoss()

    scaler = GradScaler('cuda')

    # IMPROVED: Track best model
    best_iou = 0.0

    for epoch in range(TRAIN_CONFIG["epochs"]):
        model.train()
        total_loss = 0

        # IMPROVED: Additional LR decay at epoch 40
        if epoch == 40:
            print("\n" + "="*60)
            print("FINE-TUNING: Reducing learning rate at epoch 40")
            print("="*60 + "\n")
            for g in optimizer.param_groups:
                g['lr'] = 5e-5

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']}")

        for i, (img, mask) in enumerate(pbar):
            img, mask = img.to(DEVICE), mask.to(DEVICE)

            # Debug check on first iteration
            if i == 0 and epoch == 0:
                unique_classes = torch.unique(mask)
                print(f"\nMask classes in batch: {unique_classes.tolist()}")
                print(f"Expected range: [0, {NUM_CLASSES-1}]")
                if unique_classes.max() >= NUM_CLASSES or unique_classes.min() < 0:
                    print(f"WARNING: Mask values out of range!")

            with autocast('cuda'):
                out = model(img)
                loss = criterion(out, mask) / TRAIN_CONFIG["grad_accum"]

            if torch.isnan(loss):
                print("NaN detected in loss, stopping training")
                exit()

            scaler.scale(loss).backward()

            if (i+1) % TRAIN_CONFIG["grad_accum"] == 0:
                # IMPROVED: Gradient clipping (kept from previous fixes)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss/len(train_loader)
        print(f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']} - Train Loss: {avg_loss:.4f}")

        val_iou = validate(model, val_loader)
        print(f"Epoch {epoch+1}/{TRAIN_CONFIG['epochs']} - Val IoU: {val_iou:.4f}")

        # IMPROVED: Save best model with unique name
        if val_iou > best_iou:
            best_iou = val_iou
            best_path = CHECKPOINT_DIR / f"best_model_epoch{epoch+1}_iou{val_iou:.4f}.pth"
            torch.save(model.state_dict(), best_path)
            print(f"✓ New best model saved: {best_path}")
        
        print()
    
    # IMPROVED: Save final model
    final_path = CHECKPOINT_DIR / "model_last.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Final model saved to: {final_path}")
    print(f"Best IoU achieved: {best_iou:.4f}")
    print(f"{'='*60}")


def validate(model, loader):
    model.eval()
    ious = []

    with torch.no_grad():
        for img, mask in loader:
            img, mask = img.to(DEVICE), mask.to(DEVICE)

            out = model(img)
            pred = torch.argmax(out, dim=1)

            ious.append(iou_score(pred, mask, NUM_CLASSES))

    avg_iou = sum(ious)/len(ious) if ious else 0
    return avg_iou


if __name__ == "__main__":
    train()
