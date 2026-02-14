import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
import math


class VanillaAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            heads: int,
            dim_head: int
    ):
        super().__init__()
        self.inner_dim = dim_head * heads
        self.heads = heads
        self.head_dim = dim_head
        # 保存缩放因子，用于手动计算
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim)

    def forward(self, x: torch.Tensor):
        B, L, D = x.shape

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # 保持您习惯的 einops 写法
        q = rearrange(q, "B L (h d) -> B h L d", h=self.heads)
        k = rearrange(k, "B L (h d) -> B h L d", h=self.heads)
        v = rearrange(v, "B L (h d) -> B h L d", h=self.heads)

        # 🔥🔥🔥 兼容性核心修改 🔥🔥🔥
        # 自动检测是否存在 PyTorch 2.0 的高效算子
        if hasattr(F, 'scaled_dot_product_attention'):
            # 如果有，直接用（和您原代码完全一致）
            x = F.scaled_dot_product_attention(query=q, key=k, value=v, is_causal=False)
        else:
            # 如果没有（旧版本 PyTorch），使用完全等价的数学公式手动计算
            # 1. 计算 Q * K^T / scale
            # q: [B, h, L, d], k.transpose: [B, h, d, L] -> scores: [B, h, L, L]
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

            # 2. Softmax 归一化
            attn = F.softmax(dots, dim=-1)

            # 3. 乘 V
            # attn: [B, h, L, L], v: [B, h, L, d] -> x: [B, h, L, d]
            x = torch.matmul(attn, v)

        # 还原形状
        x = rearrange(x, "B h L d -> B L (h d)")
        # 这一步通常没必要，但为了和您原代码 100% 保持一致保留下来
        x = x.to(q.dtype)

        # linear proj
        x = self.to_out(x)
        return x