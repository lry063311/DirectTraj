# DirectTraj: Real-Time and Topology-Aware Trajectory Generation via Single-Step Diffusion

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> This is the official PyTorch implementation for the paper: **"DirectTraj: Real-Time and Topology-Aware Trajectory Generation via Single-Step Diffusion"**.

## 💡 Overview

Generative modeling of trajectories is essential for high-throughput real-time background traffic rendering and city-scale digital twins. **DirectTraj** is a topology-aware generative framework based on the Diffusion Transformer (DiT) that reformulates trajectory synthesis as a **velocity-constrained direct single-step prediction mechanism**.

### Key Features:
- ⚡ **Real-Time Inference:** Compresses inference latency to an unprecedented **4 ms** (achieving a $350\times$ to $445\times$ speedup over accelerated baselines) by completely bypassing the iterative denoising paradigm.
- 🌍 **Robust Structural Scalability (Frozen Prior Transfer):** Synergizes with a pre-trained Graph Masked Autoencoder (**RoadMAE**) to extract coordinate-invariant routing rules. Using a hybrid "Frozen Prior + In-Domain Training" protocol (e.g., RoadMAE pre-trained on Porto and strictly frozen), it ensures highly robust scalability across heterogeneous city topologies like Beijing T-Drive and San Francisco.
- 🧠 **Global Receptive Field:** Replaces standard CNN U-Nets with a global DiT backbone, utilizing global self-attention for long-range sequence coherence and topological cross-attention for active road alignment.

---

## 🏗️ Model Architecture

DirectTraj consists of three synergistic phases:
1. **Dual-Stream Input Encoding:** Temporal Patching for continuous trajectories and a pre-trained RoadMAE for discrete road network graphs.
2. **Topology-Aware DiT:** Global self-attention to capture long-range routing dependencies, and topological cross-attention to actively retrieve semantic rules from the structural knowledge base.
3. **Single-Step Projection Head:** Direct mapping from unconstrained noise to the valid trajectory space in a single forward pass, optimized with explicit kinematic (velocity) consistency constraints.

---

## 🛠️ Installation & Environment Setup

We recommend using Anaconda to manage the environment.

```bash
# Create and activate conda environment
conda create -n directtraj python=3.9 -y
conda activate directtraj

# Install PyTorch (Please adjust the CUDA version to match your local machine)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)
