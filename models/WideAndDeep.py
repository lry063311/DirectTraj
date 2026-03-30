import torch
import torch.nn as nn
import torch.nn.functional as F


class WideAndDeep(nn.Module):

    def __init__(self, input_dim=5, embedding_dim=768, hidden_dim=256):
        """
        Args:
            input_dim: 输入属性的维度 (porto_HIDDEN_SIZE = 64 通常为 5)
            embedding_dim: 目标输出维度 (必须等于 DirectTraj 的 hidden_size, 即 768)
        """
        super(WideAndDeep, self).__init__()

        self.input_dim = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)  # 最终投影到 768
        )
        
    def forward(self, attr, cond_drop_prob=0.0):
        """
        Args:
            attr: [Batch, Input_Dim]
            cond_drop_prob: float, 属性 dropout 概率 (用于 Classifier-Free Guidance)
        """

        B, C = attr.shape

        if C != self.input_dim:
            if C > self.input_dim:
                x = attr[:, :self.input_dim]
            else:
                padding = torch.zeros(B, self.input_dim - C, device=attr.device)
                x = torch.cat([attr, padding], dim=1)
        else:
            x = attr

        if cond_drop_prob > 0 and self.training:
            mask = torch.bernoulli(torch.ones(B, 1, device=attr.device) * (1 - cond_drop_prob))
            x = x * mask
        return self.mlp(x)
