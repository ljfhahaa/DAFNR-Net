import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:

    p, t = prob.flatten(1), target.flatten(1)
    dice = (2 * (p * t).sum(1) + eps) / (p.sum(1) + t.sum(1) + eps)
    return 1.0 - dice.mean()


def boundary_target(mask: torch.Tensor, kernel: int = 3) -> torch.Tensor:

    pad = kernel // 2
    dil = F.max_pool2d(mask, kernel, stride=1, padding=pad)
    ero = -F.max_pool2d(-mask, kernel, stride=1, padding=pad)
    return (dil - ero).clamp_(0, 1)


class DAFNRLoss(nn.Module):
    def __init__(self, lambda_bnd: float = 0.35):
        super().__init__()
        self.lambda_bnd = lambda_bnd

    def forward(self, out: dict, mask: torch.Tensor):
        seg = out["seg"]
        l_bce = F.binary_cross_entropy_with_logits(seg, mask)
        l_dice = soft_dice_loss(torch.sigmoid(seg), mask)
        l_bnd = soft_dice_loss(torch.sigmoid(out["boundary"]), boundary_target(mask))
        total = l_bce + l_dice + self.lambda_bnd * l_bnd
        return total, {"bce": l_bce.item(), "dice": l_dice.item(), "bnd": l_bnd.item()}
