from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from .afnbem import AFNBEM
from .bggf import BGGF, cbr
from .encoder import DualStreamEncoder
class DAFNRNet(nn.Module):
    def __init__(self, num_classes: int = 1, depths: Sequence[int] = (2, 2, 6, 2), head_dim: int = 32,
                 mlp_ratio: float = 4.0, pretrained: bool = True, K: int = 8, window: int = 7,
                 sigma_par: float = 2.0, sigma_perp: float = 0.6, beta: float = 30.0, beta_w: float = 10.0,
                 c_mem: int = 64, c_b: int = 64):
        super().__init__()
        self.encoder = DualStreamEncoder(depths, head_dim, mlp_ratio, pretrained)
        c1, c2, c3, c4 = self.encoder.channels
        self.afnbem = AFNBEM(c_low=c1, c_high=c4, c_mem=c_mem, c_b=c_b, K=K, window=window,
                             sigma_par=sigma_par, sigma_perp=sigma_perp, beta=beta, beta_w=beta_w)
        self.bottleneck = cbr(c4, c4)
        self.dec3 = BGGF(c3, c4, c_b)
        self.dec2 = BGGF(c2, c3, c_b)
        self.dec1 = BGGF(c1, c2, c_b)
        self.head = nn.Conv2d(c1, num_classes, 1)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        H, W = x.shape[-2:]
        f1, f2, f3, f4 = self.encoder(x)
        fb, boundary_logits, aux = self.afnbem(f1, f4)

        d = self.bottleneck(f4)
        d, g3 = self.dec3(f3, d, fb)
        d, g2 = self.dec2(f2, d, fb)
        d, g1 = self.dec1(f1, d, fb)

        seg_logits = self.head(F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False))
        boundary_logits = F.interpolate(boundary_logits, size=(H, W), mode="bilinear", align_corners=False)
        out = {"seg": seg_logits, "boundary": boundary_logits}
        if return_aux:
            aux.update({"gates": (g1, g2, g3), "Fb": fb})
            out["aux"] = aux
        return out
