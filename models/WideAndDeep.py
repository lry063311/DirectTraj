import torch
import torch.nn as nn
import torch.nn.functional as F


class WideAndDeep(nn.Module):
    """
    通用属性编码器 (Generic Attribute Encoder)
    适配 ControlTraj 的 porto_HIDDEN_SIZE = 64 数据集头信息，并将其投影到 Transformer 的 hidden_size。
    """

    def __init__(self, input_dim=5, embedding_dim=768, hidden_dim=256):
        """
        Args:
            input_dim: 输入属性的维度 (porto_HIDDEN_SIZE = 64 通常为 5)
            embedding_dim: 目标输出维度 (必须等于 DirectTraj 的 hidden_size, 即 768)
        """
        super(WideAndDeep, self).__init__()

        self.input_dim = input_dim

        # 1. 简单的 MLP 投影
        # 将所有属性视为连续值或经过预处理的 embedding
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim)  # 最终投影到 768
        )

        # 2. 如果需要处理离散特征 (可选，视 porto_heads 具体内容而定)
        # 鉴于 ControlTraj 的预处理通常已经把时间等归一化了，直接用 MLP 是最安全的策略

    def forward(self, attr, cond_drop_prob=0.0):
        """
        Args:
            attr: [Batch, Input_Dim]
            cond_drop_prob: float, 属性 dropout 概率 (用于 Classifier-Free Guidance)
        """
        # 自动获取输入维度，防止形状不匹配
        B, C = attr.shape

        # 如果输入的维度比初始化的大或小，尝试切片或填充 (容错处理)
        if C != self.input_dim:
            if C > self.input_dim:
                x = attr[:, :self.input_dim]
            else:
                # 填充 0
                padding = torch.zeros(B, self.input_dim - C, device=attr.device)
                x = torch.cat([attr, padding], dim=1)
        else:
            x = attr

        # CFG: 随机丢弃条件 (Masking)
        if cond_drop_prob > 0 and self.training:
            # 生成 mask: 1 代表保留，0 代表丢弃
            mask = torch.bernoulli(torch.ones(B, 1, device=attr.device) * (1 - cond_drop_prob))
            x = x * mask

        # 投影
        return self.mlp(x)