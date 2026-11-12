import torch
import torch.nn as nn
import torch.nn.functional as F


def phi(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


class RALAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int = 32, dw_kernel: int = 5, eps: float = 1e-6):
        super().__init__()
        assert dim % head_dim == 0, f"dim {dim} must be divisible by head_dim {head_dim}"
        self.heads, self.d, self.eps = dim // head_dim, head_dim, eps
        self.w1 = nn.Linear(dim, dim)
        self.dwconv = nn.Conv2d(dim, dim, dw_kernel, padding=dw_kernel // 2, groups=dim)
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wg = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        u = self.w1(x).transpose(1, 2).reshape(B, C, H, W)
        u = F.silu(self.dwconv(u)).flatten(2).transpose(1, 2)
        q, k, v = self.wq(u), self.wk(u), u

        split = lambda t: t.view(B, N, self.heads, self.d).transpose(1, 2)
        q, k, v = phi(split(q)), phi(split(k)), split(v)


        q_bar = q.mean(dim=2, keepdim=True)
        logits = (q_bar @ k.transpose(-1, -2)) / self.d ** 0.5
        alpha = N * torch.softmax(logits, dim=-1)
        k = alpha.transpose(-1, -2) * k

        kv = k.transpose(-1, -2) @ v
        z = k.sum(dim=2)
        num = q @ kv
        den = (q * z.unsqueeze(2)).sum(-1, keepdim=True) + self.eps
        y = (num / den).transpose(1, 2).reshape(B, N, C)

        return self.w2(y * F.silu(self.wg(x)))


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class RALABlock(nn.Module):
    def __init__(self, dim: int, head_dim: int = 32, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim), RALAttention(dim, head_dim)
        self.norm2, self.mlp = nn.LayerNorm(dim), Mlp(dim, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = t + self.attn(self.norm1(t), H, W)
        t = t + self.mlp(self.norm2(t))
        return t.transpose(1, 2).reshape(B, C, H, W)


class RALAStage(nn.Module):


    def __init__(self, c_in: int, c_out: int, depth: int, stride: int, head_dim: int = 32, mlp_ratio: float = 4.0):
        super().__init__()
        if stride == 1 and c_in == c_out:
            self.down = nn.Identity()
        else:
            self.down = nn.Sequential(nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
                                      nn.BatchNorm2d(c_out))
        self.blocks = nn.Sequential(*[RALABlock(c_out, head_dim, mlp_ratio) for _ in range(depth)])

    def forward(self, x):
        return self.blocks(self.down(x))
