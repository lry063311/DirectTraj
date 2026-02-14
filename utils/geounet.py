"""
---
title: U-Net for ControlTraj (Full Implementation)
---
包含完整的 Attention 机制，无需外部依赖。
"""

import math
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ==============================================================================
# 1. 基础组件与注意力机制 (Integrated from unet_attention.py)
# ==============================================================================

class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        project_in = GEGLU(dim, inner_dim) if glu else nn.Linear(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.GELU() if not glu else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class CrossAttention(nn.Module):
    """
    Standard Multi-Head Attention.
    Can be used for Self-Attention (context=None) or Cross-Attention (context is not None).
    """

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        h = self.heads
        # x: [B, C, L] -> transpose to [B, L, C] for attention calculation if needed
        # But here inputs are usually [B, Sequence_Len, Dim]

        q = self.to_q(x)
        context = context if context is not None else x
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim=-1)

        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    """
    包含 Self-Attention 和 Cross-Attention 的基础块
    """

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # Self-Attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim, heads=n_heads, dim_head=d_head,
                                    dropout=dropout)  # Cross-Attention
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        # Self-Attention
        x = self.attn1(self.norm1(x)) + x
        # Cross-Attention (Geo-Attention)
        # 论文中的 Geo-Attention 将路网特征作为 context
        x = self.attn2(self.norm2(x), context=context) + x
        # Feed Forward
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    用于 UNet 的 Transformer 模块，将特征图视为序列进行处理。
    """

    def __init__(self, in_channels, n_heads, d_head, depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = normalization(in_channels)

        # 1x1 Conv to project input channels to inner_dim
        self.proj_in = nn.Conv1d(in_channels, inner_dim, kernel_size=1)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
            for _ in range(depth)
        ])

        self.proj_out = nn.Conv1d(inner_dim, in_channels, kernel_size=1)  # Restore channels

    def forward(self, x, context=None):
        # x shape: [Batch, Channels, Length]
        b, c, l = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)

        # Reshape for transformer: [Batch, Length, Channels]
        x = rearrange(x, 'b c l -> b l c')

        for block in self.transformer_blocks:
            x = block(x, context=context)

        # Reshape back: [Batch, Channels, Length]
        x = rearrange(x, 'b l c -> b c l')

        return self.proj_out(x) + x_in


# ==============================================================================
# 2. GeoUNet 主要组件
# ==============================================================================

class AttrBlock(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=256):
        super(AttrBlock, self).__init__()
        # Wide part
        self.wide_fc = nn.Linear(5, embedding_dim)
        # Deep part
        self.depature_embedding = nn.Embedding(288, hidden_dim)
        self.deep_fc1 = nn.Linear(hidden_dim, embedding_dim)
        self.deep_fc2 = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, attr):
        continuous_attrs = attr[:, 1:6]
        depature = attr[:, 0].long()
        wide_out = self.wide_fc(continuous_attrs)
        depature_embed = self.depature_embedding(depature)
        deep_out = F.relu(self.deep_fc1(depature_embed))
        deep_out = self.deep_fc2(deep_out)
        return wide_out + deep_out


class GroupNorm32(nn.GroupNorm):
    def forward(self, x): return super().forward(x.float()).type(x.dtype)


def normalization(channels): return GroupNorm32(32, channels)


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.op = nn.Conv1d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor): return self.op(x)


class ResBlock(nn.Module):
    def __init__(self, channels: int, d_t_emb: int, *, out_channels=None):
        super().__init__()
        if out_channels is None: out_channels = channels
        self.in_layers = nn.Sequential(normalization(channels), nn.SiLU(),
                                       nn.Conv1d(channels, out_channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(d_t_emb, out_channels))
        self.out_layers = nn.Sequential(normalization(out_channels), nn.SiLU(), nn.Dropout(0.),
                                        nn.Conv1d(out_channels, out_channels, 3, padding=1))
        self.skip_connection = nn.Identity() if out_channels == channels else nn.Conv1d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        h = self.in_layers(x)
        t_emb = self.emb_layers(t_emb).type(h.dtype)
        h = h + t_emb[:, :, None]
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class TimestepEmbedSequential(nn.Sequential):
    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x


class UNetModel(nn.Module):
    """
    GeoUNet Model
    """

    def __init__(self, *, in_channels: int, out_channels: int, channels: int, n_res_blocks: int,
                 attention_levels: List[int], channel_multipliers: List[int], n_heads: int, tf_layers: int = 1,
                 d_cond: int = 128):
        super().__init__()
        self.channels = channels
        levels = len(channel_multipliers)
        d_time_emb = channels * 4
        self.time_embed = nn.Sequential(nn.Linear(channels, d_time_emb), nn.SiLU(), nn.Linear(d_time_emb, d_time_emb))

        self.attr_embed = AttrBlock(embedding_dim=d_cond)

        # 💥 关键修复：特征融合层
        # 用于将 Concat(Road, Attr) 后的特征映射回 d_cond
        self.cond_fusion = nn.Linear(d_cond * 2, d_cond)

        self.input_blocks = nn.ModuleList()
        self.input_blocks.append(TimestepEmbedSequential(nn.Conv1d(in_channels, channels, 3, padding=1)))
        input_block_channels = [channels]
        channels_list = [channels * m for m in channel_multipliers]

        for i in range(levels):
            for _ in range(n_res_blocks):
                layers = [ResBlock(channels, d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                if i in attention_levels:
                    # d_head calculation: channels / n_heads
                    d_head = channels // n_heads
                    layers.append(SpatialTransformer(channels, n_heads, d_head, depth=tf_layers, context_dim=d_cond))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_channels.append(channels)
            if i != levels - 1:
                self.input_blocks.append(TimestepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)

        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels, d_time_emb),
            SpatialTransformer(channels, n_heads, channels // n_heads, depth=tf_layers, context_dim=d_cond),
            ResBlock(channels, d_time_emb),
        )

        self.output_blocks = nn.ModuleList([])
        for i in reversed(range(levels)):
            for j in range(n_res_blocks + 1):
                layers = [ResBlock(channels + input_block_channels.pop(), d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                if i in attention_levels:
                    d_head = channels // n_heads
                    layers.append(SpatialTransformer(channels, n_heads, d_head, depth=tf_layers, context_dim=d_cond))
                if i != 0 and j == n_res_blocks: layers.append(UpSample(channels))
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(normalization(channels), nn.SiLU(), nn.Conv1d(channels, out_channels, 3, padding=1))

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        half = self.channels // 2
        frequencies = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=time_steps.device)
        args = time_steps[:, None].float() * frequencies[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, cond: Optional[torch.Tensor] = None,
                attr: Optional[torch.Tensor] = None):
        """
        :param x: [B, C, L]
        :param cond: Road features [B, N_road, D_road]
        :param attr: Attributes [B, D_attr]
        """
        x_input_block = []
        t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_embed(t_emb)

        # 💥 关键修复：Concat 替代 Add，并使用 Linear 融合
        attr_emb = self.attr_embed(attr)  # [B, D]
        attr_emb_expanded = attr_emb[:, None, :].expand(-1, cond.shape[1], -1)  # [B, N_road, D]

        # Concat: [B, N_road, 2*D]
        combined_cond = torch.cat([cond, attr_emb_expanded], dim=-1)
        # Fusion: [B, N_road, D] -> 传入 SpatialTransformer 作为 context
        cond_fused = self.cond_fusion(combined_cond)

        for module in self.input_blocks:
            x = module(x, t_emb, cond_fused)
            x_input_block.append(x)

        x = self.middle_block(x, t_emb, cond_fused)

        for module in self.output_blocks:
            x = torch.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, cond_fused)

        return self.out(x)