import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from tqdm import tqdm
from einops import rearrange

# ================= 1. 导入项目模块 =================
from utils.config import config
from utils.road_encoder import MAE_ViT
from models.directtraj import DirectTraj

# ================= 2. 核心配置清单 (请务必修改这里!) =================
# 格式: "城市名": { "权重路径", "数据根目录" }
CONFIG_MAP = {
    "Porto": {
        "ckpt": r"/root/时空轨迹生成/DirectTraj/experiments/DirectTraj_porto_20260113_121606/models/checkpoint_last.pt",
        "data_root": r"/root/时空轨迹生成/DirectTraj/data/porto_HIDDEN_SIZE = 64",
        "save_name": "porto_generated.npy"
    },
    "T-Drive": {  # 或者是 Beijing
        "ckpt": r"/root/时空轨迹生成/DirectTraj/experiments/DirectTraj_beijing_20260113_143040/models/checkpoint_last.pt",
        # 🔥 替换为你的北京权重路径
        "data_root": r"/root/时空轨迹生成/DirectTraj/data/beijing",
        "save_name": "beijing_generated.npy"
    },
    "SF": {
        "ckpt": r"/root/时空轨迹生成/DirectTraj/experiments/DirectTraj_sanfrancisco_20260113_202820/models/checkpoint_last.pt",
        # 🔥 替换为你的SF权重路径
        "data_root": r"/root/时空轨迹生成/DirectTraj/data/sanfrancisco",
        "save_name": "sanfrancisco_generated.npy"
    }
}

OUTPUT_DIR = r"/root/DirectTraj/experiments/all_cities_gen"
BATCH_SIZE = 1024
DEVICE = config.DEVICE


# ================= 3. 辅助组件 =================
class RoadMAEProcessor:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(RoadMAEProcessor, cls).__new__(cls)
            cls._instance.pool = nn.AvgPool1d(kernel_size=2, stride=2).to(DEVICE)
            cls._instance._feat_adapt_layer = None
        return cls._instance

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
            road_emb = F.interpolate(road_emb.permute(0, 2, 1), size=config.Model.TARGET_COND_LEN, mode='linear',
                                     align_corners=False).permute(0, 2, 1)

        if road_emb.shape[-1] != config.Model.D_COND:
            if self._feat_adapt_layer is None:
                self._feat_adapt_layer = nn.Linear(road_emb.shape[-1], config.Model.D_COND).to(DEVICE)
            road_emb = self._feat_adapt_layer(road_emb)
        return road_emb


def one_shot_sampling(model, x_noise, road_emb, attr):
    max_t = config.Diffusion.NUM_DIFFUSION_TIMESTEPS - 1
    t = torch.full((x_noise.shape[0],), max_t, device=DEVICE, dtype=torch.long)
    pred_x0 = model(x_noise, t, attr, road_emb)
    pred_x0 = torch.clamp(pred_x0, -5.0, 5.0)
    return pred_x0


def load_stats(json_path):
    with open(json_path, 'r') as f:
        stats = json.load(f)
    mu = np.array([stats['lon_mean'], stats['lat_mean']])
    std = np.array([stats['lon_std'], stats['lat_std']])
    return torch.from_numpy(mu).float().to(DEVICE).view(1, 1, 2), \
        torch.from_numpy(std).float().to(DEVICE).view(1, 1, 2)


# ================= 4. 主逻辑 =================
def run_inference(city_name, cfg):
    print(f"\n{'=' * 20} Processing {city_name} {'=' * 20}")

    # 1. 构建路径
    data_root = cfg['data_root']
    # 自动识别文件名变体
    city_tag = os.path.basename(data_root)  # e.g., 'porto_HIDDEN_SIZE = 64'

    traj_path = os.path.join(data_root, f"{city_tag}_trajs.npy")
    road_path = os.path.join(data_root, f"{city_tag}_roads.npy")
    attr_path = os.path.join(data_root, f"{city_tag}_heads.npy")
    stats_path = os.path.join(data_root, f"{city_tag}_global_stats.json")

    # 容错: T-Drive 有时叫 beijing
    if not os.path.exists(traj_path) and city_name == "T-Drive":
        traj_path = os.path.join(data_root, "beijing_trajs.npy")
        road_path = os.path.join(data_root, "beijing_roads.npy")
        attr_path = os.path.join(data_root, "beijing_heads.npy")
        stats_path = os.path.join(data_root, "beijing_global_stats.json")

    # 2. 加载数据
    print(f"📥 Loading data from: {data_root}")
    try:
        trajs = torch.from_numpy(np.load(traj_path)).float()
        heads = torch.from_numpy(np.load(attr_path)).float()
        roads = torch.from_numpy(np.load(road_path)).float()
    except Exception as e:
        print(f"❌ Failed to load data for {city_name}: {e}")
        return

    # 维度修正
    if trajs.ndim == 3 and trajs.shape[1] != 2: trajs = trajs.permute(0, 2, 1)

    # 切分测试集
    np.random.seed(config.DATA_SPLIT_SEED)
    total_len = len(trajs)
    test_idx = np.random.permutation(total_len)[:int(total_len * 0.2)]

    test_ds = torch.utils.data.TensorDataset(trajs[test_idx], heads[test_idx], roads[test_idx])
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 3. 加载统计信息 (用于可能的归一化)
    mu, std = load_stats(stats_path)

    # 4. 加载模型
    print(f"🧠 Loading Checkpoint: {cfg['ckpt']}")
    road_encoder = MAE_ViT(
        image_size=config.Data.TRAJ_LENGTH, patch_size=5, emb_dim=config.Model.D_COND,
        encoder_layer=config.Model.MAE_ENCODER_LAYERS, encoder_head=4, decoder_layer=4, decoder_head=4,
        mask_ratio=config.Model.MAE_MASK_RATIO
    ).to(DEVICE)

    model = DirectTraj(
        traj_length=config.Data.TRAJ_LENGTH, hidden_size=config.Model.HIDDEN_SIZE, depth=config.Model.DEPTH,
        num_heads=config.Model.NUM_HEADS, mlp_ratio=config.Model.MLP_RATIO, cond_dim=config.Model.D_COND,
        lon_lat_embedding=True
    ).to(DEVICE)

    checkpoint = torch.load(cfg['ckpt'], map_location=DEVICE)
    if 'road_encoder_state_dict' in checkpoint: road_encoder.load_state_dict(checkpoint['road_encoder_state_dict'])
    if 'ema_shadow' in checkpoint:
        for name, param in model.named_parameters():
            if name in checkpoint['ema_shadow']: param.data.copy_(checkpoint['ema_shadow'][name].data)
    else:
        model.load_state_dict(checkpoint.get('unet_state_dict', checkpoint))

    model.eval();
    road_encoder.eval()
    road_processor = RoadMAEProcessor()

    # 5. 推理循环
    generated_trajs = []
    print("🚀 Generating...")

    with torch.no_grad():
        for batch in tqdm(test_loader):
            attr = batch[1].to(DEVICE)
            road_raw = batch[2].to(DEVICE)

            # 🔥🔥🔥 [智能归一化检测] 🔥🔥🔥
            # 检查 road 数据的第一段均值。如果绝对值 > 10，说明是原始坐标，必须归一化！
            check_val = road_raw[0, :, :2].mean().abs().item()

            road_input = road_raw.clone()
            if check_val > 5.0:
                # 执行归一化
                road_input[..., :2] = (road_raw[..., :2] - mu) / std

            # 提取特征
            road_mae_input = road_processor.prepare_road_mae_input(road_input)
            road_emb = road_processor.extract_road_features(road_encoder, road_mae_input, no_mask=True)

            # 采样
            x_T = torch.randn(attr.shape[0], config.Data.CHANNELS, config.Data.TRAJ_LENGTH, device=DEVICE)
            x0_gen = one_shot_sampling(model, x_T, road_emb, attr)

            generated_trajs.append(x0_gen.cpu().numpy())

    # 6. 保存
    all_gen = np.concatenate(generated_trajs, axis=0)
    save_path = os.path.join(OUTPUT_DIR, cfg['save_name'])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(save_path, all_gen)
    print(f"✅ Saved {city_name} results to: {save_path} (Shape: {all_gen.shape})")


def main():
    for city, cfg in CONFIG_MAP.items():
        # 简单检查权重文件是否存在
        if not os.path.exists(cfg['ckpt']):
            print(f"⚠️ Checkpoint missing for {city}, skipping...")
            continue
        run_inference(city, cfg)

    print("\n🎉 All cities processed! Now run 'viz_heatmap_3x2.py'.")


if __name__ == "__main__":
    main()