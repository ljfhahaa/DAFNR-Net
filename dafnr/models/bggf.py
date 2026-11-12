import torch
import torch.nn as nn
import torch.nn.functional as F


def cbr(c_in: int, c_out: int, k: int = 3) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class UpProject(nn.Module):


    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.proj = cbr(c_in, c_out, k=1)

    def forward(self, x: torch.Tensor, size) -> torch.Tensor:
        return self.proj(F.interpolate(x, size=size, mode="bilinear", align_corners=False))


class BGGF(nn.Module):
    def __init__(self, c_skip: int, c_dec: int, c_b: int):
        super().__init__()
        self.up = UpProject(c_dec, c_skip)
        self.gate = nn.Sequential(nn.Conv2d(c_b, 1, 1, bias=False), nn.BatchNorm2d(1))
        self.fuse = cbr(c_skip, c_skip)

    def forward(self, f_enc: torch.Tensor, f_dec_prev: torch.Tensor, fb: torch.Tensor):
        size = f_enc.shape[-2:]
        f_dec = self.up(f_dec_prev, size)
        fb_i = F.interpolate(fb, size=size, mode="bilinear", align_corners=False)
        g = torch.sigmoid(self.gate(fb_i))
        return self.fuse(g * f_enc + (1.0 - g) * f_dec), g
