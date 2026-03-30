import torch

# 🔥 [DEBUG] 强制开启 TF32 (针对 3090/4090 显卡加速)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"🔥 [Env] TF32 Acceleration Enabled: {torch.backends.cuda.matmul.allow_tf32}")

import torch.nn as nn
import numpy as np
import math
import datetime
import os
import warnings
import json
import random
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace

# 🚀 屏蔽干扰警告
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="timm")

import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from einops import rearrange
from torch.cuda.amp import autocast, GradScaler
from utils.config import config
from utils.road_encoder import MAE_ViT
from models.directtraj import DirectTraj

class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel): module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad: self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel): module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema_copy(self, module):
        module_copy = deepcopy(module)
        if isinstance(module_copy, nn.DataParallel): module_copy = module_copy.module
        for name, param in module_copy.named_parameters():
            if param.requires_grad: param.data.copy_(self.shadow[name].data)
        return module_copy

class RoadMAEProcessor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RoadMAEProcessor, cls).__new__(cls)
            cls._instance.pool = nn.AvgPool1d(kernel_size=2, stride=2).to(config.DEVICE)
            cls._instance._feat_adapt_layer = None
        return cls._instance

    def reset_adapt_layer(self):
        self._feat_adapt_layer = None

    def prepare_road_mae_input(self, road_segments):
        return road_segments[..., :2].permute(0, 2, 1)

    def extract_road_features(self, road_encoder, road_mae_input, no_mask=False):
        road_encoder.eval()
        with torch.no_grad():
            features, _ = road_encoder.encoder(road_mae_input, no_mask=no_mask)

        global_feat = features[0:1]
        local_feats = features[1:]
        global_feat_expanded = global_feat.expand(local_feats.shape[0], -1, -1)
        fused_feats = local_feats + global_feat_expanded
        road_emb = rearrange(fused_feats, 't b c -> b t c')

        if road_emb.shape[1] != config.Model.TARGET_COND_LEN:
            road_emb = F.interpolate(
                road_emb.permute(0, 2, 1),
                size=config.Model.TARGET_COND_LEN,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)

        if road_emb.shape[-1] != config.Model.D_COND:
            if self._feat_adapt_layer is None:
                self._feat_adapt_layer = nn.Linear(road_emb.shape[-1], config.Model.D_COND).to(config.DEVICE)
            road_emb = self._feat_adapt_layer(road_emb)
        return road_emb

class Logger:
    def __init__(self, log_file):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def info(self, msg): self._log("INFO", msg)

    def error(self, msg): self._log("ERROR", msg)

    def warn(self, msg): self._log("WARN", msg)

    def _log(self, level, msg):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        content = f"[{timestamp}] [{level}] {msg}"
        print(content)
        with open(self.log_file, "a", encoding="utf-8") as f: f.write(content + "\n")

def set_seed(seed=config.DATA_SPLIT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed()

def gather(consts: torch.Tensor, t: torch.Tensor):
    return consts.gather(dim=-1, index=t.clamp(0, consts.shape[-1] - 1)).reshape(-1, 1, 1)

def setup_diffusion_parameters():
    num_steps = config.Diffusion.NUM_DIFFUSION_TIMESTEPS
    beta_start = config.Diffusion.BETA_START
    beta_end = config.Diffusion.BETA_END
    if config.Diffusion.BETA_SCHEDULE == 'cosine':
        betas = [beta_start + (beta_end - beta_start) * (1 - math.cos(i / num_steps * math.pi)) / 2 for i in
                 range(num_steps)]
    elif config.Diffusion.BETA_SCHEDULE == 'linear':
        betas = torch.linspace(beta_start, beta_end, num_steps)
    else:
        raise ValueError(f"Unknown beta schedule: {config.Diffusion.BETA_SCHEDULE}")
    beta = torch.tensor(betas, device=config.DEVICE, dtype=torch.float32)
    alpha = 1. - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar

beta, alpha, alpha_bar = setup_diffusion_parameters()

def load_data():
    try:
        trajs = torch.from_numpy(np.load(config.Data.TRAJ_PATH)).float()
        heads = torch.from_numpy(np.load(config.Data.ATTR_PATH)).float()
        roads = torch.from_numpy(np.load(config.Data.ROAD_PATH)).float()

        if trajs.ndim == 3 and trajs.shape[1] != 2:
            trajs = trajs.permute(0, 2, 1)

        print(f"📊 Loading stats from: {config.Data.STATS_PATH}")
        with open(config.Data.STATS_PATH, 'r') as f:
            global_stats = json.load(f)

        min_len = min(len(trajs), len(heads), len(roads))
        return trajs[:min_len], heads[:min_len], roads[:min_len], global_stats
    except Exception as e:
        raise RuntimeError(f"数据加载失败: {e}")

def get_data_split_indices(trajs, test_ratio=0.2, seed=config.DATA_SPLIT_SEED):
    np.random.seed(seed)
    total_len = len(trajs)
    indices = np.random.permutation(total_len)
    n_test = int(total_len * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    print(f"\n✂️ [实验设置对齐] 数据集切分 (Seed={seed})")
    print(f"   - 总量: {total_len}")
    print(f"   - 训练集 (80%): {len(train_idx)}")
    print(f"   - 测试集 (20%): {len(test_idx)} <--- 评估样本量")
    return train_idx, test_idx


def init_models():
    road_encoder = MAE_ViT(
        image_size=config.Data.TRAJ_LENGTH,
        patch_size=5,
        emb_dim=config.Model.D_COND,
        encoder_layer=config.Model.MAE_ENCODER_LAYERS,
        encoder_head=config.Model.NUM_HEADS,
        decoder_layer=4,
        decoder_head=4,
        mask_ratio=config.Model.MAE_MASK_RATIO
    ).to(config.DEVICE)

    model = DirectTraj(
        traj_length=config.Data.TRAJ_LENGTH,
        hidden_size=config.Model.HIDDEN_SIZE,
        depth=config.Model.DEPTH,
        num_heads=config.Model.NUM_HEADS,
        mlp_ratio=config.Model.MLP_RATIO,
        cond_dim=config.Model.D_COND,
        lon_lat_embedding=True
    ).to(config.DEVICE)

    target_filename = "road_encoder.pt"
    pretrained_path = config.MODEL_ROOT / target_filename

    if not pretrained_path.exists():
        raise FileNotFoundError(f"❌ 错误：在 {config.MODEL_ROOT} 下未找到必须的权重文件 '{target_filename}'！")

    try:
        ckpt = torch.load(pretrained_path, map_location=config.DEVICE)
        road_encoder.load_state_dict(ckpt.get('model_state_dict', ckpt))
        print(f"✅ [Success] RoadMAE 预训练权重加载成功: {pretrained_path}")
    except Exception as e:
        raise RuntimeError(f"❌ 加载权重文件失败: {e}")

    for param in road_encoder.parameters(): param.requires_grad = False
    road_encoder.eval()
    return road_encoder, model

def setup_experiment_directories():
    base_name = f"DirectTraj_{config.Data.DATASET}"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = config.EXPERIMENT_ROOT / f"{base_name}_{timestamp}"

    dirs = SimpleNamespace(
        models=exp_dir / "models",
        results=exp_dir / "results",
        logs=exp_dir / "logs",
        visualizations=exp_dir / "visualizations",
        code_backup=exp_dir / "code_backup"
    )

    for d in [dirs.models, dirs.logs, dirs.results]:
        d.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config_snapshot.txt", "w") as f:
        f.write(f"Dataset: {config.Data.DATASET}\n")
        f.write(f"Batch Size: {config.Training.BATCH_SIZE}\n")
        f.write(f"LR: {config.Training.LR}\n")

    return dirs, exp_dir

def save_checkpoint(epoch, model, road_encoder, optimizer, scheduler, ema_helper, metrics_history, best_density_err,
                    path, global_step):
    state = {
        'epoch': epoch, 'unet_state_dict': model.state_dict(), 'road_encoder_state_dict': road_encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
        'ema_shadow': ema_helper.shadow, 'metrics_history': metrics_history,
        'best_density_err': best_density_err, 'global_step': global_step
    }
    torch.save(state, path)
    checkpoint_dir = Path(path).parent
    all_checkpoints = sorted(list(checkpoint_dir.glob("checkpoint_epoch_*.pt")),
                             key=lambda x: int(x.stem.split('_')[-1]))
    if len(all_checkpoints) > config.MAX_CHECKPOINTS_TO_KEEP:
        for old_ckpt in all_checkpoints[:-config.MAX_CHECKPOINTS_TO_KEEP]: old_ckpt.unlink()


def load_checkpoint(checkpoint_path, model, road_encoder, optimizer, scheduler, ema_helper, logger):
    if not Path(checkpoint_path).exists(): return 0, [], [], float('inf'), 0
    try:
        checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['unet_state_dict'])
        if 'road_encoder_state_dict' in checkpoint: road_encoder.load_state_dict(checkpoint['road_encoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if ema_helper and 'ema_shadow' in checkpoint:
            if not hasattr(ema_helper, 'shadow'): ema_helper.register(model)
            ema_helper.shadow = checkpoint['ema_shadow']
        start_epoch = checkpoint.get('epoch', 0) + 1
        return start_epoch, [], checkpoint.get('metrics_history', []), checkpoint.get('best_density_err',
                                                                                      float('inf')), checkpoint.get(
            'global_step', 0)
    except Exception as e:
        logger.error(f"❌ 加载失败：{str(e)}")
        return 0, [], [], float('inf'), 0

def train_model(resume_from=config.RESUME_EXPERIMENT):
    logger = None
    try:
        if resume_from and Path(resume_from).exists():
            exp_dir = Path(resume_from)
            dirs = SimpleNamespace(models=exp_dir / "models", results=exp_dir / "results", logs=exp_dir / "logs",
                                   visualizations=exp_dir / "visualizations", code_backup=exp_dir / "code_backup")
            logger = Logger(dirs.logs / "train.log")
            logger.info(f"🔄 恢复训练：{exp_dir}")
        else:
            dirs, exp_dir = setup_experiment_directories()
            logger = Logger(dirs.logs / "train.log")
            logger.info("🚀 开始新训练 (DirectTraj)")
            resume_from = None

        full_trajs, full_heads, full_roads, porto_stats = load_data()

        def auto_tune_params(trajs, sample_size=1000):
            indices = torch.randperm(len(trajs))[:min(len(trajs), sample_size)]
            sample = trajs[indices]
            if sample.shape[1] == 2: sample = sample.permute(0, 2, 1)

            diffs = sample[:, 1:, :] - sample[:, :-1, :]
            step_dist = torch.norm(diffs, dim=-1).mean().item()

            logger.info(f"📏 [AutoTune] 测得归一化平均步长: {step_dist:.4f}")

            if step_dist > 0.05:
                return 5.0, "Large Scale"
            else:
                return 1.0, "Small Scale"

        vel_weight, mode_name = auto_tune_params(full_trajs)
        logger.info(f"⚙️ [AutoTune] 模式: {mode_name} | 自动设定 vel_weight = {vel_weight}")

        train_idx, test_idx = get_data_split_indices(full_trajs, test_ratio=0.2)

        train_ds = TensorDataset(full_trajs[train_idx], full_heads[train_idx], full_roads[train_idx])
        test_ds = TensorDataset(full_trajs[test_idx], full_heads[test_idx], full_roads[test_idx])

        train_dl = DataLoader(train_ds, batch_size=config.Training.BATCH_SIZE, shuffle=True, drop_last=True,
                              num_workers=config.Data.NUM_WORKERS,pin_memory=True,prefetch_factor=2,persistent_workers=True)
        test_dl = DataLoader(test_ds, batch_size=config.Sampling.BATCH_SIZE, shuffle=False,
                             num_workers=config.Data.NUM_WORKERS,pin_memory=True,prefetch_factor=2,persistent_workers=True)

        road_encoder, model = init_models()
        if int(torch.__version__.split('.')[0]) >= 2:
            print("🚀 [PyTorch 2.0] Enabling torch.compile for massive speedup...")
            try:
                model = torch.compile(model, mode='default')
            except Exception as e:
                print(f"⚠️ torch.compile failed: {e}")
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.Training.LR,
                                      weight_decay=config.Training.WEIGHT_DECAY)
        scheduler = CosineAnnealingLR(optimizer, T_max=config.Training.N_EPOCHS,
                                      eta_min=config.Training.SCHEDULER_PARAMS["eta_min"])
        ema_helper = EMAHelper(mu=config.Model.EMA_RATE)
        ema_helper.register(model)
        road_processor = RoadMAEProcessor()
        scaler = GradScaler(enabled=config.Training.USE_AMP)

        start_epoch, _, metrics_history, best_density_err, global_step = 0, [], [], float('inf'), 0
        if resume_from:
            latest = dirs.models / "checkpoint_last.pt"
            if latest.exists():
                start_epoch, _, metrics_history, best_density_err, global_step = load_checkpoint(latest, model,
                                                                                                 road_encoder,
                                                                                                 optimizer, scheduler,
                                                                                                 ema_helper, logger)

        total_steps = len(train_dl) * config.Training.N_EPOCHS
        warmup_steps = int(total_steps * config.Training.WARMUP_RATIO)
        if start_epoch > 0 and global_step > warmup_steps:
            for _ in range(start_epoch): scheduler.step()

        for epoch in range(start_epoch, config.Training.N_EPOCHS):
            model.train()
            epoch_loss = 0.0
            acc_steps = 0
            pbar = tqdm(train_dl, desc=f"Epoch {epoch}")

            for batch in pbar:
                global_step += 1
                if global_step <= warmup_steps:
                    lr_scale = float(global_step) / warmup_steps
                    for pg in optimizer.param_groups: pg['lr'] = config.Training.LR * lr_scale

                x0 = batch[0].to(config.DEVICE, dtype=torch.float32)
                attr = batch[1].to(config.DEVICE, dtype=torch.float32)
                road = batch[2].to(config.DEVICE, dtype=torch.float32)

                road_in = road_processor.prepare_road_mae_input(road)
                road_emb = road_processor.extract_road_features(road_encoder, road_in, no_mask=False)

                if np.random.random() < 0.1: road_emb = None

                t = torch.randint(0, config.Diffusion.NUM_DIFFUSION_TIMESTEPS, (x0.shape[0],), device=config.DEVICE)
                noise = torch.randn_like(x0)
                x_t = gather(torch.sqrt(alpha_bar), t) * x0 + gather(torch.sqrt(1 - alpha_bar), t) * noise

                with autocast(enabled=config.Training.USE_AMP):
                    pred_x0 = model(x_t, t, attr, road_emb)
                    loss_pos = F.mse_loss(pred_x0, x0)
                    loss_vel = F.mse_loss(pred_x0[..., 1:] - pred_x0[..., :-1], x0[..., 1:] - x0[..., :-1])
                    loss = (loss_pos + vel_weight * loss_vel) / config.Training.GRADIENT_ACCUMULATION_STEPS

                scaler.scale(loss).backward()
                epoch_loss += loss.item() * config.Training.GRADIENT_ACCUMULATION_STEPS
                acc_steps += 1

                if acc_steps % config.Training.GRADIENT_ACCUMULATION_STEPS == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.Training.CLIP_GRAD_NORM)
                    scaler.step(optimizer);
                    scaler.update();
                    optimizer.zero_grad();
                    ema_helper.update(model)
                    acc_steps = 0

                pbar.set_postfix(loss=f"{loss.item() * config.Training.GRADIENT_ACCUMULATION_STEPS:.4f}")

            if global_step > warmup_steps: scheduler.step()
            logger.info(f"Epoch {epoch:3d} | Avg Loss: {epoch_loss / len(train_dl):.6f}")

            if (epoch + 1) % config.Training.SNAPSHOT_FREQ == 0:
                save_checkpoint(epoch, model, road_encoder, optimizer, scheduler, ema_helper, metrics_history,
                                best_density_err, dirs.models / f"checkpoint_epoch_{epoch}.pt", global_step)

            save_checkpoint(epoch, model, road_encoder, optimizer, scheduler, ema_helper, metrics_history,
                            best_density_err, dirs.models / "checkpoint_last.pt", global_step)

    except KeyboardInterrupt:
        print("Training interrupted.")
    except Exception as e:
        if logger:
            logger.error(f"Error: {e}")
        else:
            print(f"Error: {e}")
        raise e


if __name__ == "__main__":
    train_model()
