import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class PatchShuffle(nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor, no_mask=False):
        # 💥 修复：增加 no_mask 参数，在推理或作为条件使用时保留所有 Patch
        if no_mask or self.ratio == 0:
            T, B, C = patches.shape
            # 不进行 Shuffle 和 Mask，直接返回原序
            # 构造虚假的索引以保持接口一致
            forward_indexes = torch.arange(T, device=patches.device).unsqueeze(1).repeat(1, B)
            backward_indexes = torch.argsort(forward_indexes, dim=0)
            return patches, forward_indexes, backward_indexes

        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes


class MAE_Encoder(nn.Module):
    def __init__(self, image_size=400, patch_size=5, emb_dim=192, num_layer=12, num_head=3, mask_ratio=0.75):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        # 修复位置编码尺寸计算：确保整除或向上取整
        num_patches = image_size // patch_size
        self.pos_embedding = nn.Parameter(torch.zeros(num_patches, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = nn.Conv1d(2, emb_dim, patch_size, patch_size)
        self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img, no_mask=False):
        # img: (B, 2, L) -> (B, D, N)
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c l -> l b c')

        # 加上位置编码
        # 注意：这里假设输入长度固定，如果长度变化需要切片 pos_embedding
        patches = patches + self.pos_embedding

        # 💥 关键修复：传入 no_mask 标志
        patches, forward_indexes, backward_indexes = self.shuffle(patches, no_mask=no_mask)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes


class MAE_Decoder(nn.Module):
    def __init__(self, image_size=200, patch_size=5, emb_dim=192, num_layer=4, num_head=2):
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.zeros((image_size // patch_size) + 1, 1, emb_dim))
        self.transformer = nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.head = nn.Linear(emb_dim, 2 * patch_size)
        self.patch2img = Rearrange('h b (c p) -> b c (h p)', p=patch_size, h=image_size // patch_size)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat(
            [torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat(
            [features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)],
            dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T - 1:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)
        return img, mask


class MAE_ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=2, emb_dim=192, encoder_layer=12, encoder_head=3, decoder_layer=4,
                 decoder_head=3, mask_ratio=0.75):
        super().__init__()
        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img, no_mask=False):
        # 💥 传递 no_mask 参数
        features, backward_indexes = self.encoder(img, no_mask=no_mask)
        if no_mask: return features  # 如果不遮蔽，直接返回特征用于 Condition

        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask