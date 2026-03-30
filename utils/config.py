import torch
import platform
import os
from pathlib import Path


class Config:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    IS_LINUX = platform.system() == 'Linux'
    class Data:
        DATASET = 'porto'
        _utils_dir = Path(__file__).resolve().parent
        _project_dir = _utils_dir.parent
        CURRENT_DATA_DIR = _project_dir / "data" / DATASET

        print(f"[Config] Dataset selected: {DATASET}")
        print(f"[Config] Data Directory:   {CURRENT_DATA_DIR}")

        if platform.system() == 'Windows':
            NUM_WORKERS = 0
        else:
            NUM_WORKERS = 16

        TRAJ_PATH = CURRENT_DATA_DIR / f"{DATASET}_trajs.npy"
        ROAD_PATH = CURRENT_DATA_DIR / f"{DATASET}_roads.npy"
        ATTR_PATH = CURRENT_DATA_DIR / f"{DATASET}_heads.npy"

        STATS_PATH = CURRENT_DATA_DIR / f"{DATASET}_global_stats.json"

        TRAJ_LENGTH = 200
        CHANNELS = 2
    class Model:
        HIDDEN_SIZE = 128
        DEPTH = 6
        NUM_HEADS = 4
        MLP_RATIO = 4.0
        D_COND = 128
        TARGET_COND_LEN = 200
        MAE_ENCODER_LAYERS = 8

        MAE_MASK_RATIO = 0.25

        EMA_RATE = 0.999
        EMA = True

    class Diffusion:
        BETA_SCHEDULE = 'linear'
        BETA_START = 0.0001
        BETA_END = 0.02

        NUM_DIFFUSION_TIMESTEPS = 500

        DIFFUSION_SKIP_STEPS = 5

    class Training:
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

    class Sampling:
        BATCH_SIZE = 128
        PAPER_EVAL_STEPS = 500
        
    DATA_ROOT = Path("./data")
    MODEL_ROOT = Path("./models")
    EXPERIMENT_ROOT = Path("./experiments")

    MAX_CHECKPOINTS_TO_KEEP = 3
    DATA_SPLIT_SEED = 42

    RESUME_EXPERIMENT = None


config = Config()
