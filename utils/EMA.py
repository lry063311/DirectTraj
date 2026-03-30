import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from .config import config
from .geounet import UNetModel

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

        if isinstance(inner_module, UNetModel):
            unet_args = {
                'in_channels': config.Model.IN_CHANNELS,
                'out_channels': config.Model.OUT_CHANNELS,
                'channels': config.Model.CHANNELS,
                'n_res_blocks': config.Model.NUM_RES_BLOCKS,
                'attention_levels': config.Model.ATTENTION_LEVELS,
                'channel_multipliers': config.Model.CHANNEL_MULTIPLIERS,
                'n_heads': config.Model.N_HEADS,
                'tf_layers': config.Model.TF_LAYERS,
                'd_cond': config.Model.D_COND
            }
            module_copy = UNetModel(**unet_args).to(config.DEVICE)
            module_copy.load_state_dict(inner_module.state_dict())

        else:
            module_copy = deepcopy(module)

        if is_dataparallel:
            module_copy = nn.DataParallel(module_copy)

        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict
