import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config as cfg

class BCELoss(nn.Module):
    """ Standard Binary Cross-Entropy Loss with logits. """
    def __init__(self, pos_weight=None):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)

class DiceLoss(nn.Module):
    """ Standard Dice Loss. """
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        B = targets.size(0)
        p_flat = probs.view(B, -1)
        t_flat = targets.view(B, -1)
        intersection = (p_flat * t_flat).sum(1)
        dice_score = (2.0 * intersection + self.smooth) / (p_flat.sum(1) + t_flat.sum(1) + self.smooth)
        return 1 - dice_score.mean()

class FocalLoss(nn.Module):
    """ Standard Focal Loss. """
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()

        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss

        if self.alpha is not None:
            alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()

class TverskyLoss(nn.Module):
    """ Tversky Loss for fine-tuning precision-recall trade-off. """
    def __init__(self, alpha=None, smooth=1e-5):
        super().__init__()
        if alpha is None:
            alpha = getattr(cfg, 'tversky_alpha', 0.25)
        self.alpha = alpha
        self.beta = 1 - alpha
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        targets = targets.float()
        B = targets.size(0)
        p_flat = probs.view(B, -1)
        t_flat = targets.view(B, -1)
        tp = (p_flat * t_flat).sum(1)
        fp = ((1 - t_flat) * p_flat).sum(1)
        fn = (t_flat * (1 - p_flat)).sum(1)
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky_index.mean()

