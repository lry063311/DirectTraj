import torch
import platform
import os
from pathlib import Path


class Config:
    """
    DirectTraj 论文复现最终严格版配置 (Strict Reproduction - DiT Version)
    兼容: Windows (Local Debug) & Linux (4090 Training)
    """

    # -------------------------- 1. 硬件配置 --------------------------
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 自动识别是否为 Linux 系统
    IS_LINUX = platform.system() == 'Linux'

    # -------------------------- 2. 数据配置 --------------------------
    class Data:
        DATASET = 'porto'

        # 1. 自动定位项目根目录
        _utils_dir = Path(__file__).resolve().parent
        _project_dir = _utils_dir.parent

        # 2. 定位到具体的子文件夹: data/chengdu 或 data/porto_HIDDEN_SIZE = 64
        CURRENT_DATA_DIR = _project_dir / "data" / DATASET

        print(f"[Config] Dataset selected: {DATASET}")
        print(f"[Config] Data Directory:   {CURRENT_DATA_DIR}")

        # --- 环境适配 (保持不变) ---
        if platform.system() == 'Windows':
            NUM_WORKERS = 0
        else:
            NUM_WORKERS = 16

        # 3. 自动拼接文件名 (根据你预处理保存的命名规则 f"{dataset}_trajs.npy")
        TRAJ_PATH = CURRENT_DATA_DIR / f"{DATASET}_trajs.npy"
        ROAD_PATH = CURRENT_DATA_DIR / f"{DATASET}_roads.npy"
        ATTR_PATH = CURRENT_DATA_DIR / f"{DATASET}_heads.npy"

        # 🔥 新增：把统计文件的路径也放在配置里，main.py 就不用猜了
        STATS_PATH = CURRENT_DATA_DIR / f"{DATASET}_global_stats.json"

        TRAJ_LENGTH = 200
        CHANNELS = 2
    # -------------------------- 3. 模型参数 (DiT Backbone) --------------------------
    class Model:
        HIDDEN_SIZE = 128 # 超参数分析实验一
        DEPTH = 6
        NUM_HEADS = 4
        MLP_RATIO = 4.0
        D_COND = 128
        TARGET_COND_LEN = 200
        MAE_ENCODER_LAYERS = 8

        # 🔥🔥🔥 [关键修改 1] Mask Ratio 黄金区间：0.25
        # 既防止死记硬背(Overfitting)，又保证能看清路网(Conditioning)
        MAE_MASK_RATIO = 0.25

        EMA_RATE = 0.999
        EMA = True

    # -------------------------- 4. 扩散参数 --------------------------
    class Diffusion:
        BETA_SCHEDULE = 'linear'
        BETA_START = 0.0001

        # 🔥🔥🔥 [关键修改 2] 降低噪声上限：0.05 -> 0.02
        # 解决“心电图”式的乱麻轨迹，让生成结果更平滑
        BETA_END = 0.02

        NUM_DIFFUSION_TIMESTEPS = 500 # 超参数分析实验二

        # 🟢【训练用】拆分为 100 步进行精细学习
        DIFFUSION_SKIP_STEPS = 5

    # -------------------------- 5. 训练配置 (针对 4090 优化) --------------------------
    class Training:
        # 🔥 4090 策略: 物理 BS 128 * 累积 8 = 等效 BS 1024
        BATCH_SIZE = 256
        GRADIENT_ACCUMULATION_STEPS = 4

        N_EPOCHS = 200

        LR = 2e-4
        WEIGHT_DECAY = 1e-4

        USE_AMP = True
        CLIP_GRAD_NORM = 1.0
        WARMUP_RATIO = 0.05

        SCHEDULER = "CosineAnnealingLR"
        SCHEDULER_PARAMS = {
            "T_max": 200,
            "eta_min": 1e-6
        }

        SNAPSHOT_FREQ = 20
        VALIDATION_FREQ = 10

    # -------------------------- 6. 采样配置 --------------------------
    class Sampling:
        BATCH_SIZE = 128

        # 🔴【论文评估用】One-Shot 极速推理 (500步直接跳到底)
        PAPER_EVAL_STEPS = 500

    # -------------------------- 7. 路径与日志 --------------------------
    DATA_ROOT = Path("./data")
    MODEL_ROOT = Path("./models")
    EXPERIMENT_ROOT = Path("./experiments")

    MAX_CHECKPOINTS_TO_KEEP = 3
    DATA_SPLIT_SEED = 42

    # 🔥🔥🔥 [检查] 恢复训练路径 (必须是目录)
    # 如果是新训练，保持 None；如果要恢复，填入'/root/时空轨迹生成/DirectTraj/experiments/DirectTraj_porto_20260113_121606'
    RESUME_EXPERIMENT = None


config = Config()