import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def oriented_granules(K: int = 8, window: int = 7, sigma_par: float = 2.0, sigma_perp: float = 0.6) -> torch.Tensor:

    r = window // 2
    dy, dx = torch.meshgrid(torch.arange(-r, r + 1, dtype=torch.float64),
                            torch.arange(-r, r + 1, dtype=torch.float64), indexing="ij")
    dx, dy = dx.reshape(-1), dy.reshape(-1)
    kernels = []
    for k in range(K):
        theta = k * math.pi / K

        d_par = dx * math.cos(theta) + dy * math.sin(theta)
        d_perp = -dx * math.sin(theta) + dy * math.cos(theta)
        kernels.append(torch.exp(-d_par ** 2 / (2 * sigma_par ** 2) - d_perp ** 2 / (2 * sigma_perp ** 2)))
    return torch.stack(kernels).float()


def soft_max(z: torch.Tensor, beta: float, dim: int) -> torch.Tensor:

    return torch.logsumexp(beta * z, dim=dim) / beta


def soft_min(z: torch.Tensor, beta: float, dim: int) -> torch.Tensor:

    return -torch.logsumexp(-beta * z, dim=dim) / beta


class FuzzyRoughDescriptor(nn.Module):

    def __init__(self, K=8, window=7, sigma_par=2.0, sigma_perp=0.6, beta=30.0, beta_w=10.0, pad_mode="replicate"):
        super().__init__()
        assert window % 2 == 1, "window size must be odd"
        self.K, self.window, self.beta, self.beta_w, self.pad_mode = K, window, beta, beta_w, pad_mode
        self.register_buffer("R", oriented_granules(K, window, sigma_par, sigma_perp), persistent=False)
        theta = torch.arange(K, dtype=torch.float32) * math.pi / K
        self.register_buffer("cos2", torch.cos(2 * theta), persistent=False)
        self.register_buffer("sin2", torch.sin(2 * theta), persistent=False)

    def responses(self, X: torch.Tensor) -> torch.Tensor:

        B, _, H, W = X.shape
        r = self.window // 2
        Xp = F.pad(X, (r, r, r, r), mode=self.pad_mode)
        Xn = F.unfold(Xp, kernel_size=self.window).view(B, 1, -1, H, W)
        R = self.R.view(1, self.K, -1, 1, 1)
        lower = soft_min(torch.maximum(1.0 - R, Xn), self.beta, dim=2)
        upper = soft_max(torch.minimum(R, Xn), self.beta, dim=2)
        return upper - lower

    def forward(self, X: torch.Tensor):
        E = self.responses(X)
        A = E.amax(dim=1, keepdim=True) - E.amin(dim=1, keepdim=True)
        w = torch.softmax(-self.beta_w * E, dim=1)
        c = (w * self.cos2.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        s = (w * self.sin2.view(1, -1, 1, 1)).sum(dim=1, keepdim=True)
        return E, A, c, s

    @staticmethod
    def tangent(c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:

        return 0.5 * torch.atan2(s, c)


class AFNBEM(nn.Module):

    def __init__(self, c_low=64, c_high=512, c_mem=64, c_b=64, **descriptor_kwargs):
        super().__init__()
        self.membership_head = nn.Sequential(
            nn.Conv2d(c_low + c_high, c_mem, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_mem),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mem, 1, 1),
        )
        self.descriptor = FuzzyRoughDescriptor(**descriptor_kwargs)

        self.proj = nn.Sequential(
            nn.Conv2d(3, c_b, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_b),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_b, c_b, 1),
        )
        self.boundary_head = nn.Conv2d(c_b, 1, 1)

    def forward(self, f1: torch.Tensor, f4: torch.Tensor):
        f4_up = F.interpolate(f4, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        X = torch.sigmoid(self.membership_head(torch.cat([f1, f4_up], dim=1)))
        E, A, c, s = self.descriptor(X)
        Fb = self.proj(torch.cat([A, c, s], dim=1))
        boundary_logits = self.boundary_head(Fb)
        aux = {"membership": X, "E": E, "A": A, "c": c, "s": s}
        return Fb, boundary_logits, aux
