import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from timm.models.vision_transformer import Mlp
from models.attention import VanillaAttention
from models.patchify import PatchEmbed1D, PatchEmbed2D
from models.pos_emb import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid
from models.WideAndDeep import WideAndDeep


# -----------------------------------------------------------------------------
# 创新点 3: 拓扑感知交叉注意力模块 (Topology-Aware Cross-Attention)
# -----------------------------------------------------------------------------
class TopologyCrossAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        # Q: 轨迹特征, K/V: 路网拓扑特征
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, traj_tokens, road_features):
        """
        Args:
            traj_tokens (Query): [Batch, Seq_Len, Dim]
            road_features (Key/Value): [Batch, Num_Roads, Dim]
        """
        # 🔥 如果路网被 Dropout (传入 None)，直接返回原特征 (即退化为纯 DiT)
        if road_features is None:
            return traj_tokens

        # Q-K-V 交互：轨迹 Token 动态查询相关的路网特征
        attn_output, _ = self.cross_attn(query=traj_tokens,
                                         key=road_features,
                                         value=road_features)
        # 残差连接 + LayerNorm
        return self.norm(traj_tokens + attn_output)


# -----------------------------------------------------------------------------
# 辅助模块: Timestep Embedding
# -----------------------------------------------------------------------------
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# -----------------------------------------------------------------------------
# DirectTraj Block: 融合了 Cross-Attention 的 DiT Block
# -----------------------------------------------------------------------------
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DirectTrajBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        # 1. Self-Attention
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = VanillaAttention(hidden_size, num_heads, hidden_size // num_heads)

        # 2. [创新点] Cross-Attention
        self.cross_attn = TopologyCrossAttention(hidden_size, num_heads)

        # 3. FFN
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)

        # 4. adaLN
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, road_features):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = self.cross_attn(x, road_features)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# -----------------------------------------------------------------------------
# 主模型: DirectTraj
# -----------------------------------------------------------------------------
class DirectTraj(nn.Module):
    def __init__(
            self,
            traj_length=200,
            hidden_size=256,
            depth=6,
            num_heads=4,
            mlp_ratio=4.0,
            cond_dim=256,
            lon_lat_embedding=True
    ):
        super().__init__()
        self.out_channels = 1 if lon_lat_embedding else 2
        self.patch_size = 1
        self.traj_length = traj_length
        self.lon_lat_embedding = lon_lat_embedding
        self.hidden_size = hidden_size

        # 1. 输入嵌入
        if self.lon_lat_embedding:
            self.x_embedder = PatchEmbed2D(embed_dim=hidden_size)
        else:
            self.x_embedder = PatchEmbed1D(embed_dim=hidden_size)

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = WideAndDeep(embedding_dim=hidden_size)

        seq_len = 2 * traj_length if lon_lat_embedding else traj_length
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, hidden_size), requires_grad=False)

        # 2. 维度适配
        self.road_projector = nn.Linear(cond_dim, hidden_size)

        # 3. 骨干网络
        self.blocks = nn.ModuleList([
            DirectTrajBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # 4. 输出层
        self.final_layer = FinalLayer(hidden_size, self.patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = None
        if self.lon_lat_embedding:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.traj_length)
        else:
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1],
                                                          np.arange(self.traj_length, dtype=np.float32))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.lon_lat_embedding:
            w_lon = self.x_embedder.proj_lon.weight.data
            nn.init.xavier_uniform_(w_lon.view([w_lon.shape[0], -1]))
            w_lat = self.x_embedder.proj_lat.weight.data
            nn.init.xavier_uniform_(w_lat.view([w_lat.shape[0], -1]))

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x):
        x_lon, x_lat = x[:, 0::2, :], x[:, 1::2, :]
        trajs = torch.cat((x_lon, x_lat), dim=2)
        trajs = rearrange(trajs, 'B L C -> B C L').contiguous()
        return trajs

    def forward(self, x, t, y, road_features=None):
        """
        Args:
            x: [B, 2, L] 含噪轨迹
            t: [B] 时间步
            y: [B, ...] 属性
            road_features: [B, N, D] 路网特征 (可能为 None)
        """
        x = self.x_embedder(x) + self.pos_embed
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, cond_drop_prob=0.1 if self.training else 0.0)
        c = t_emb + y_emb

        # 🔥🔥🔥 [关键修复]：只在 road_features 存在时进行投影 🔥🔥🔥
        # 否则如果 road_features 是 None (Condition Dropout), road_projector 会报错
        if road_features is not None:
            road_features = self.road_projector(road_features)

        for block in self.blocks:
            x = block(x, c, road_features)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        return x