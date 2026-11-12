from typing import List, Sequence

import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

from .rala import RALAStage


class ConvGate(nn.Module):


    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.gate = nn.Sequential(
            nn.Conv2d(2 * channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, 3, padding=1),
        )

    def forward(self, c: torch.Tensor, t: torch.Tensor):
        g = torch.sigmoid(self.gate(torch.cat([c, t], dim=1)))
        return g * c + (1.0 - g) * t, g


class DualStreamEncoder(nn.Module):
    def __init__(self, depths: Sequence[int] = (2, 2, 6, 2), head_dim: int = 32, mlp_ratio: float = 4.0,
                 pretrained: bool = True):
        super().__init__()
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        r = resnet34(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.cnn = nn.ModuleList([r.layer1, r.layer2, r.layer3, r.layer4])
        self.channels: List[int] = [64, 128, 256, 512]
        c_in = [64] + self.channels[:-1]
        self.rala = nn.ModuleList([
            RALAStage(c_in[i], self.channels[i], depths[i], stride=1 if i == 0 else 2,
                      head_dim=head_dim, mlp_ratio=mlp_ratio)
            for i in range(4)
        ])
        self.cg = nn.ModuleList([ConvGate(c) for c in self.channels])

    def forward(self, x: torch.Tensor, return_gates: bool = False):
        c = t = self.stem(x)
        feats, gates = [], []
        for cnn_stage, rala_stage, cg in zip(self.cnn, self.rala, self.cg):
            c, t = cnn_stage(c), rala_stage(t)
            f, g = cg(c, t)
            feats.append(f)
            gates.append(g)
            c, t = c + f, t + f
        return (feats, gates) if return_gates else feats
