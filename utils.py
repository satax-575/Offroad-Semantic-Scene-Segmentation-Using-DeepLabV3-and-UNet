import torch
import torch.nn as nn
import numpy as np


class DiceLoss(nn.Module):
    def forward(self, pred, target):
        pred = torch.softmax(pred, dim=1)
        target_onehot = torch.zeros_like(pred)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)

        inter = (pred * target_onehot).sum()
        union = pred.sum() + target_onehot.sum()

        return 1 - (2 * inter + 1) / (union + 1)


class FocalLoss(nn.Module):
    """IMPROVED: Added Focal Loss for hard example mining"""
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def lovasz_softmax(probs, labels):
    """FIXED: Numerically stable Lovasz loss"""
    C = probs.shape[1]
    losses = []

    for c in range(C):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue

        pred = probs[:, c]
        err = (fg - pred).abs()
        
        # FIXED: Flatten before sorting
        err = err.view(-1)
        fg = fg.view(-1)
        
        if err.numel() == 0:
            continue

        err_sorted, idx = torch.sort(err, descending=True)
        fg_sorted = fg[idx]

        # FIXED: Add epsilon for numerical stability
        grad = torch.cumsum(fg_sorted, 0) / (fg_sorted.sum() + 1e-6)
        losses.append(torch.dot(err_sorted, grad))

    # FIXED: Handle empty losses case
    if len(losses) == 0:
        return torch.tensor(0.0, device=probs.device)
    
    return torch.mean(torch.stack(losses))


class CompetitionLoss(nn.Module):
    """DEPRECATED: Old loss function - kept for compatibility"""
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss()
        self.dice = DiceLoss()

    def forward(self, logits, target):
        probs = torch.softmax(logits, dim=1)
        # FIXED: Adjusted weights to prevent loss explosion
        focal_loss = self.focal(logits, target)
        dice_loss = self.dice(logits, target)
        lovasz_loss = lovasz_softmax(probs, target)
        
        return (
            0.5 * focal_loss +
            0.4 * dice_loss +
            0.1 * lovasz_loss
        )


class BetterLoss(nn.Module):
    """IMPROVED: Better loss function with class weights"""
    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, logits, target):
        return (
            0.4 * self.ce(logits, target) +
            0.3 * self.dice(logits, target) +
            0.3 * self.focal(logits, target)
        )


def iou_score(pred, target, n_classes):
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    ious = []
    for cls in range(n_classes):
        p = pred == cls
        t = target == cls

        if t.sum() == 0:
            continue

        inter = (p & t).sum()
        union = (p | t).sum()

        ious.append(inter / union if union else 0)

    return np.mean(ious) if ious else 0
