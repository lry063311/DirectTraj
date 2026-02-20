# DirectTraj: Real-time and Topology-Aware Urban Trajectory Generation via Direct Diffusion

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> This is the official PyTorch implementation for the paper: **"DirectTraj: Real-time and Topology-Aware Urban Trajectory Generation via Direct Diffusion"**.

## 💡 Overview

Generative modeling of urban trajectories often struggles with the trade-off between structural fidelity and inference speed. **DirectTraj** is a knowledge-driven generative framework based on the Diffusion Transformer (DiT) that reformulates trajectory synthesis as a **Physics-Informed One-Shot Manifold Projection**. 

### Key Features:
- ⚡ **Real-Time Inference:** Compresses inference latency to **4 ms** (a $350\times \sim 445\times$ speedup over iterative SOTA baselines) via a Just-in-Time (JiT) non-iterative projection.
- 🌍 **Zero-Shot Generalization:** Utilizes a Graph Masked Autoencoder (**RoadMAE**) to extract coordinate-independent topological rules, enabling robust transfer to unseen cities (e.g., trained on Porto, tested on Beijing T-Drive).
- 🧠 **Global Receptive Field:** Replaces standard CNN U-Nets with a Topology-Aware DiT, strictly preserving long-range spatial coherence and Origin-Destination (OD) constraints.

---

## 🏗️ Model Architecture

DirectTraj consists of three synergistic phases:
1. **Dual-Stream Input Encoding:** Temporal Patching for trajectories and RoadMAE for road network graphs.
2. **Topology-Aware DiT:** Global self-attention for sequence coherence and topological cross-attention for active road alignment.
3. **JiT Projection Head:** Direct mapping from Gaussian noise to the valid trajectory manifold in a single step.

---

## 🛠️ Installation & Environment Setup

We recommend using Anaconda to manage the environment.

```bash
# Create and activate conda environment
conda create -n directtraj python=3.9 -y
conda activate directtraj

# Install PyTorch (Please adjust the CUDA version to match your local machine
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
