import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# -------------------------- 【关键修复点 1：导入所需依赖】 --------------------------
# 必须导入全局 config 才能访问 nested class 中的参数
# 假设 config 在上层目录，使用相对导入
from .config import config
# 必须导入 UNetModel 才能正确判断和实例化
from .geounet import UNetModel


# ----------------------------------------------------------------------------------

class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (
                                                 1. -
                                                 self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        """
        创建模型的EMA副本。
        核心修复：针对 UNetModel，手动传递所有必需的关键字参数，解决 TypeError。
        """
        is_dataparallel = isinstance(module, nn.DataParallel)
        inner_module = module.module if is_dataparallel else module

        # -------------------------- 【关键修复点 2：手动实例化 UNetModel】 --------------------------
        if isinstance(inner_module, UNetModel):
            # 从全局配置中提取 UNetModel 所需的所有关键字参数
            unet_args = {
                'in_channels': config.Model.IN_CHANNELS,
                'out_channels': config.Model.OUT_CHANNELS,
                'channels': config.Model.CHANNELS,
                'n_res_blocks': config.Model.NUM_RES_BLOCKS,
                'attention_levels': config.Model.ATTENTION_LEVELS,
                'channel_multipliers': config.Model.CHANNEL_MULTIPLIERS,
                'n_heads': config.Model.N_HEADS,
                'tf_layers': config.Model.TF_LAYERS,
                # 注意：d_cond 是 UNetModel 构造函数中第8个必需的关键字参数
                'd_cond': config.Model.D_COND
            }

            # 使用正确的关键字参数重新实例化 UNetModel
            module_copy = UNetModel(**unet_args).to(config.DEVICE)
            module_copy.load_state_dict(inner_module.state_dict())

        else:
            # 对于其他模型，使用传统的深度拷贝
            module_copy = deepcopy(module)

        # ------------------------------------------------------------------------------------------

        if is_dataparallel:
            module_copy = nn.DataParallel(module_copy)

        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict