import torch
import torch.nn as nn
from einops import rearrange
from timm.layers.helpers import to_2tuple

class PatchEmbed1D(nn.Module):
    """1D Traj to 1D Embedding"""
    def __init__(
        self,
        traj_length=200,
        patch_size=1,
        in_chans=2,
        embed_dim=768,
        norm_layer=None,
    ):
        super().__init__()
        assert traj_length % patch_size == 0, "traj_length must be divisible by patch_size"
        self.out_length = traj_length // patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        x = self.proj(x)
        assert x.shape[2] == self.out_length, f"Input length ({L}) doesn't match model ({self.out_length})."
        x = rearrange(x, "B C L -> B L C")
        x = self.norm(x)
        return x
    
class PatchEmbed2D(nn.Module):
    """1D Traj to 2D Embedding"""
    def __init__(
        self,
        traj_length=200,
        patch_size=1,
        embed_dim=768,
        norm_layer=None,
    ):
        super().__init__()
        assert traj_length % patch_size == 0, "traj_length must be divisible by patch_size"
        self.out_length = (traj_length // patch_size) * 2
        self.proj_lon = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.proj_lat = nn.Conv1d(1, embed_dim, kernel_size=patch_size, stride=patch_size, padding=0)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        x_lon, x_lat = x[:, 0:1, :], x[:, 1:2, :]
        x_lon_emb, x_lat_emb = self.proj_lon(x_lon), self.proj_lat(x_lat) # B, C, L
        x_lon_emb, x_lat_emb = rearrange(x_lon_emb, "B C L -> B C 1 L"), rearrange(x_lat_emb, "B C L -> B C 1 L") # B, C, 1, L
        x_emb = torch.cat((x_lon_emb, x_lat_emb), dim=2) # B, C, 2, L

        x_emb = rearrange(x_emb, "B C H W -> B W H C")
        x_emb = rearrange(x_emb, "B W H C -> B (W H) C")
        assert x_emb.shape[1] == self.out_length
        x_emb = self.norm(x_emb)
        return x_emb