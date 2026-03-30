import torch
import numpy as np
import time
from scipy.stats import entropy
from haversine import haversine
from tqdm import tqdm
from einops import rearrange


class TrajEvaluator:
    def __init__(self, config, global_stats, device):
        self.cfg = config
        self.stats = global_stats
        self.device = device

        self.gt_trajs = None
        self.fixed_bounds = None
        self.gt_density_hist = None
        self.gt_trip_hist = None
        self.gt_len_hist = None

        self.dynamic_limit = 50.0

    def initialize(self, test_loader, road_processor=None):
        print("📊 [Evaluator] 初始化：正在加载测试集并计算固定边界...")
        self.road_processor = road_processor

        all_real = []
        for batch in test_loader:
            x0 = batch[0].to(self.device)

            real_denorm = self._denormalize(x0)
            all_real.append(real_denorm.cpu().numpy().transpose(0, 2, 1))

        self.gt_trajs = np.concatenate(all_real, axis=0)

        all_pts = self.gt_trajs.reshape(-1, 2)
        all_pts = all_pts[~np.isnan(all_pts).any(axis=1)]
        ds_name = str(self.cfg.Data.DATASET).lower()
        if 'porto_HIDDEN_SIZE = 64' in ds_name:
            print(f"   🎯 检测到 Porto 数据集：启用 Tight Mode (Padding=0, Percentile=0.5-99.5)")
            pct_range = [0.5, 99.5]
            padding = 0.0
        else:
            print(f"   🛡️ 检测到 {self.cfg.Data.DATASET} 数据集：启用 Safe Mode (Padding=0.02, Percentile=0-100)")
            pct_range = [0.1, 99.9]
            padding = 0.02
        lon_min, lon_max = np.percentile(all_pts[:, 0], pct_range)
        lat_min, lat_max = np.percentile(all_pts[:, 1], pct_range)
        self.fixed_bounds = {
            "lon_bound": [lon_min - padding, lon_max + padding],
            "lat_bound": [lat_min - padding, max(lat_max, lat_min + 0.01) + padding]
        }
        print(f"   ✅ 边界已锁定: {self.fixed_bounds}")

        raw_lens = self._get_len_raw_values(self.gt_trajs, use_mask=True)
        p99 = np.percentile(raw_lens, 99)
        self.dynamic_limit = np.ceil(p99 / 10) * 10
        print(f"   📏 [AutoTune] GT 99% 长度: {p99:.2f} km")
        print(f"   ⚙️ [AutoTune] 自动设定评估截断值 (Clip Limit): {self.dynamic_limit} km")

        self.gt_density_hist = self._to_grid(self.gt_trajs, self.fixed_bounds)
        gt_trip_pts = np.stack([self.gt_trajs[:, 0, :], self.gt_trajs[:, -1, :]], axis=1)
        self.gt_trip_hist = self._to_grid(gt_trip_pts, self.fixed_bounds)
        self.gt_len_hist = self._get_len_dist(self.gt_trajs, use_mask=True)
        print("   ✅ GT 分布计算完毕")

    def _denormalize(self, x):
        x = x.clone().detach()
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        x[:, 0, :] = x[:, 0, :] * self.stats["lon_std"] + self.stats["lon_mean"]
        x[:, 1, :] = x[:, 1, :] * self.stats["lat_std"] + self.stats["lat_mean"]

        x[:, 0, :] = torch.clamp(x[:, 0, :], -180.0, 180.0)
        x[:, 1, :] = torch.clamp(x[:, 1, :], -90.0, 90.0)
        return x

    def _get_valid_length_mask(self, trajs):
        masks = []
        for t in trajs:
            diffs = np.linalg.norm(t[1:] - t[:-1], axis=1)

            is_moving = diffs > 1e-6

            valid_len = len(t)
            for i in range(len(t) - 2, 0, -1):
                if is_moving[i]:
                    valid_len = i + 2
                    break

            m = np.zeros(len(t), dtype=bool)
            m[:valid_len] = True
            masks.append(m)
        return masks

    def _get_len_raw_values(self, trajs, use_mask=True):
        vals = []
        is_beijing = 'beijing' in str(self.cfg.Data.DATASET).lower()
        if use_mask:
            masks = self._get_valid_length_mask(trajs)
        else:
            masks = [np.ones(len(t), dtype=bool) for t in trajs]

        for i, t in enumerate(trajs):
            valid_t = t[masks[i]]

            if len(valid_t) < 2:
                vals.append(0.0)
                continue

            d = 0.0
            for k in range(len(valid_t) - 1):
                step = haversine((valid_t[k, 1], valid_t[k, 0]),
                                 (valid_t[k + 1, 1], valid_t[k + 1, 0]))
                if is_beijing:
                    d += step
                else:
                    if step <= 1.0: d += step
            vals.append(d)
        return vals

    def _get_len_dist(self, trajs, use_mask=True):
        raw_lens = self._get_len_raw_values(trajs, use_mask=use_mask)
        clipped = np.clip(raw_lens, 0.0, self.dynamic_limit)
        hist, _ = np.histogram(clipped, bins=np.linspace(0, self.dynamic_limit, 51))
        return hist.astype(float) / (hist.sum() + 1e-10)

    def _to_grid(self, trajs, bounds, grid_num=16):
        lon_min, lon_max = bounds["lon_bound"]
        lat_min, lat_max = bounds["lat_bound"]
        grid = np.zeros((grid_num, grid_num))

        if trajs.ndim == 3:
            all_pts = np.concatenate(trajs, axis=0)
        else:
            all_pts = trajs

        all_pts = all_pts[~np.isnan(all_pts).any(axis=1)]

        mask = (all_pts[:, 0] >= lon_min) & (all_pts[:, 0] <= lon_max) & \
               (all_pts[:, 1] >= lat_min) & (all_pts[:, 1] <= lat_max)
        valid = all_pts[mask]

        c = ((valid[:, 0] - lon_min) / (lon_max - lon_min) * grid_num).astype(int)
        r = ((valid[:, 1] - lat_min) / (lat_max - lat_min) * grid_num).astype(int)

        np.add.at(grid, (np.clip(r, 0, grid_num - 1), np.clip(c, 0, grid_num - 1)), 1)
        return grid.flatten() / (grid.sum() + 1e-10)

    def _js_divergence(self, p, q):
        m = 0.5 * (p + q)
        return 0.5 * (entropy(p, m) + entropy(q, m))
    @torch.no_grad()
    def evaluate(self, model, road_encoder, test_loader, sampling_func):
        model.eval()
        if road_encoder is not None:
            road_encoder.eval()

        gen_trajs = []
        print("⏱️  [Evaluator] 正在执行单条推理测速 (BS=1)...")

        dummy_batch = next(iter(test_loader))
        x0_single = dummy_batch[0][0:1].to(self.device)
        attr_single = dummy_batch[1][0:1].to(self.device)
        road_single = dummy_batch[2][0:1].to(self.device) if len(dummy_batch) > 2 else None

        r_emb_single = None
        if road_encoder is not None and road_single is not None:
            r_in = self.road_processor.prepare_road_mae_input(road_single)
            r_emb_single = self.road_processor.extract_road_features(road_encoder, r_in, no_mask=True)
        elif hasattr(model, 'config') and hasattr(model.config, 'model'):
            r_emb_single = torch.zeros((1, 1, model.config.model.ch)).to(self.device)

        x_noise_single = torch.randn_like(x0_single)

        # 预热
        for _ in range(5):
            _ = sampling_func(model, x_noise_single, r_emb_single, attr_single)

        # 测速
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        latencies = []

        for _ in range(50):
            torch.cuda.synchronize()
            start_event.record()
            _ = sampling_func(model, x_noise_single, r_emb_single, attr_single)
            end_event.record()
            torch.cuda.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

        avg_latency = np.mean(latencies)
        print(f"⏱️  单条推理延迟 (Real-time Latency): {avg_latency:.4f} ms")

        print("🚀 [Evaluator] 正在生成全量测试集 (Auto-Masking Enabled)...")

        for batch in tqdm(test_loader, desc="Generating"):
            x0_real = batch[0].to(self.device)
            attr = batch[1].to(self.device)
            road = batch[2].to(self.device) if len(batch) > 2 else None

            r_emb = None
            if road_encoder is not None and road is not None:
                r_in = self.road_processor.prepare_road_mae_input(road)
                r_emb = self.road_processor.extract_road_features(road_encoder, r_in, no_mask=True)
            elif hasattr(model, 'config') and hasattr(model.config, 'model'):
                r_emb = torch.zeros((len(x0_real), 1, model.config.model.ch)).to(self.device)

            x_noise = torch.randn_like(x0_real)

            # 生成
            x_gen = sampling_func(model, x_noise, r_emb, attr)

            # 反归一化
            gen_np = self._denormalize(x_gen).cpu().numpy().transpose(0, 2, 1)
            gen_trajs.append(gen_np)

        gen_all = np.concatenate(gen_trajs, axis=0)
        print("📊 [Evaluator] 计算指标中...")

        d_err = self._js_divergence(self.gt_density_hist, self._to_grid(gen_all, self.fixed_bounds))

        gen_trip_pts = np.stack([gen_all[:, 0, :], gen_all[:, -1, :]], axis=1)
        t_err = self._js_divergence(self.gt_trip_hist, self._to_grid(gen_trip_pts, self.fixed_bounds))

        l_err = self._js_divergence(self.gt_len_hist, self._get_len_dist(gen_all, use_mask=True))
        real_len_raw = self._get_len_raw_values(self.gt_trajs, use_mask=True)
        gen_len_raw = self._get_len_raw_values(gen_all, use_mask=True)

        print("\n" + "=" * 40)
        print("🕵️‍♂️ [DEBUG] 结果概览:")
        print(f"   ⏱️ Latency: {avg_latency:.2f} ms")
        print(f"   📏 Length Error: {l_err:.4f}")
        print(f"   🤖 Gen Avg Length: {np.mean(gen_len_raw):.4f} km (GT: {np.mean(real_len_raw):.4f})")
        print("=" * 40 + "\n")
        lon_mean = self.stats['lon_mean']

        if lon_mean > 100:  # 北京经度约 116
            dataset_name = "beijing"
        elif lon_mean < -100:  # 旧金山经度约 -122
            dataset_name = "sf"
        else:  # 波尔图经度约 -8
            dataset_name = "porto_HIDDEN_SIZE = 64"

        print(f"🌍 [Auto-Detect] 检测到数据集: {dataset_name} (Lon: {lon_mean:.1f})")
        save_name_ours = f"{dataset_name}_ours.npy"
        save_name_gt = f"{dataset_name}_gt.npy"
        np.save(save_name_ours, np.array(gen_len_raw))
        np.save(save_name_gt, np.array(real_len_raw))

        print(f"💾 [Auto-Save] DirectTraj (Ours) Saved: {save_name_ours}")
        print(f"💾 [Auto-Save] Ground Truth Saved: {save_name_gt}")

        return {
            "Density Error": d_err,
            "Trip Error": t_err,
            "Length Error": l_err,
            "Latency (ms)": avg_latency
        }
