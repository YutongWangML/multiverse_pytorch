import torch
import torch.nn as nn
class DKRLoss(nn.Module):
    def __init__(self, num_classes, reduction = 'mean'):
        super(DKRLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        K = self.num_classes
        z = y_pred
        y = y_true
        z_at_y = z[torch.arange(z.shape[0]), y]
        ell_inv = 1 / (torch.arange(K) + 1)
        z_sorted = torch.sort(z, axis=1, descending=True).values
        z_sorted_cumsum = z_sorted.cumsum(axis=1)
        losses = 1 - z_at_y + torch.max(ell_inv*z_sorted_cumsum - ell_inv , axis=1).values
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'none':
            return losses
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")
