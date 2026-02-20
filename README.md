#DirectTraj：基于直接扩散的实时且拓扑感知的城市轨迹生成

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![许可证：MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

>这是论文《DirectTraj：基于直接扩散的实时且拓扑感知的城市轨迹生成》的官方PyTorch实现。

##概述

城市轨迹的生成建模通常难以在结构保真度与推理速度之间取得平衡。DirectTraj是一种基于扩散Transformer（DiT）的知识驱动生成框架，它将轨迹合成重新表述为一种**基于物理的一次性流形投影。

###关键特性：
通过即时（JiT）非迭代投影，将推理延迟压缩至4毫秒（相较于迭代式SOTA基准测试，速度提升约350倍至445倍）。
-零样本泛化：利用图掩码自编码器（RoadMAE)提取与坐标无关的拓扑规则，从而实现对未见城市的稳健迁移（例如，在波尔图训练，于北京T-Drive测试）。
- 🧠 全局感受野：用拓扑感知的DiT取代标准CNN U-Net，严格保持长程空间连贯性和起讫点（OD）约束。

---

##️ 安装与环境配置

我们建议使用Anaconda来管理环境。

```bash

# 创建并激活conda环境
conda create -n directtraj python=3.9 -y
conda activate directtraj

# 安装PyTorch（请根据本地机器调整CUDA版本，例如cu118）
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
