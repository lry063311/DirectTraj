import numpy as np
import torch

# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py
def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, traj_length):
    pos_spatio = np.arange(2, dtype=np.float32)
    pos_temporal = np.arange(traj_length, dtype=np.float32)
    pos_emb_spatio = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, pos_spatio)
    pos_emb_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, pos_temporal)
    assert pos_emb_spatio.shape == (2, embed_dim // 2) and pos_emb_temporal.shape == (traj_length, embed_dim // 2)
    # concat to 2D positional embedding.
    pos_emb_spatio = torch.from_numpy(pos_emb_spatio).repeat(traj_length, 1)
    pos_emb_temporal = torch.from_numpy(pos_emb_temporal.repeat(2, axis=0))
    assert pos_emb_spatio.shape == pos_emb_temporal.shape and pos_emb_spatio.shape == (2*traj_length, embed_dim // 2)
    pos_emb_2D = torch.cat([pos_emb_spatio, pos_emb_temporal], dim=1)
    assert pos_emb_2D.shape == (2*traj_length, embed_dim)
    return pos_emb_2D.numpy()

if __name__ == "__main__":
    pos_emb2D = get_2d_sincos_pos_embed(768, 200)
    print(pos_emb2D.shape)