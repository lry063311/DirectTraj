import torch
import numpy as np
import time
from scipy.stats import entropy
from haversine import haversine
from tqdm import tqdm
from einops import rearrange


class TrajEvaluator:
    def __init__(self, config, global_stats, device):
        """
        [终极通用版评估器] - v4.0
        1. 兼容性: 智能识别 DataLoader 返回长度 (2或3)，支持 DiffTraj, ControlTraj, Traj-Transformer
        2. 修复: 针对 Porto 等短轨迹数据集，内置 Mask 机制剔除尾部噪声
        3. 鲁棒性: 物理截断防止坐标溢出，自适应计算截断阈值
        """
        self.cfg = config
        self.stats = global_stats
        self.device = device

        # 缓存 GT 数据
        self.gt_trajs = None
        self.fixed_bounds = None
        self.gt_density_hist = None
        self.gt_trip_hist = None
        self.gt_len_hist = None

        # 动态截断值
        self.dynamic_limit = 50.0

    def initialize(self, test_loader, road_processor=None):
        print("📊 [Evaluator] 初始化：正在加载测试集并计算固定边界...")
        self.road_processor = road_processor

        # 1. 收集所有真实轨迹 (GT)
        all_real = []
        for batch in test_loader:
            # 🔥 [兼容性修复] 智能解包：不再强制解包为3个变量
            # batch[0] 始终是轨迹 x0
            x0 = batch[0].to(self.device)

            real_denorm = self._denormalize(x0)
            all_real.append(real_denorm.cpu().numpy().transpose(0, 2, 1))

        self.gt_trajs = np.concatenate(all_real, axis=0)

        # 2. 计算全城固定边界
        all_pts = self.gt_trajs.reshape(-1, 2)
        all_pts = all_pts[~np.isnan(all_pts).any(axis=1)]
        # 获取数据集名称 (转小写)
        ds_name = str(self.cfg.Data.DATASET).lower()
        if 'porto_HIDDEN_SIZE = 64' in ds_name:
            # 🟢 Porto 模式：极致收紧，为了让 Density 不为 0
            print(f"   🎯 检测到 Porto 数据集：启用 Tight Mode (Padding=0, Percentile=0.5-99.5)")
            pct_range = [0.5, 99.5]
            padding = 0.0
        else:
            # 🔵 SF / Beijing 模式：安全模式，防止切掉边缘热点 (机场/高速口)
            print(f"   🛡️ 检测到 {self.cfg.Data.DATASET} 数据集：启用 Safe Mode (Padding=0.02, Percentile=0-100)")
            # 恢复到全范围，防止 SF 的边缘点被切
            pct_range = [0.1, 99.9]
            padding = 0.02
        lon_min, lon_max = np.percentile(all_pts[:, 0], pct_range)
        lat_min, lat_max = np.percentile(all_pts[:, 1], pct_range)
        self.fixed_bounds = {
            "lon_bound": [lon_min - padding, lon_max + padding],
            "lat_bound": [lat_min - padding, max(lat_max, lat_min + 0.01) + padding]
        }
        print(f"   ✅ 边界已锁定: {self.fixed_bounds}")

        # 3. 🔥 [Mask 增强] 计算 GT 长度分布
        # 注意：这里开启 use_mask=True，确保 GT 里的 0-padding 被正确忽略
        raw_lens = self._get_len_raw_values(self.gt_trajs, use_mask=True)
        p99 = np.percentile(raw_lens, 99)
        self.dynamic_limit = np.ceil(p99 / 10) * 10
        print(f"   📏 [AutoTune] GT 99% 长度: {p99:.2f} km")
        print(f"   ⚙️ [AutoTune] 自动设定评估截断值 (Clip Limit): {self.dynamic_limit} km")

        # 4. 预计算 GT 直方图
        self.gt_density_hist = self._to_grid(self.gt_trajs, self.fixed_bounds)
        # Trip Error 取起点和终点 (Mask 逻辑保证了终点的有效性)
        gt_trip_pts = np.stack([self.gt_trajs[:, 0, :], self.gt_trajs[:, -1, :]], axis=1)
        self.gt_trip_hist = self._to_grid(gt_trip_pts, self.fixed_bounds)
        self.gt_len_hist = self._get_len_dist(self.gt_trajs, use_mask=True)
        print("   ✅ GT 分布计算完毕")

    # ================= 核心工具函数 =================

    def _denormalize(self, x):
        """反归一化 + 物理截断 (防止 NaN)"""
        x = x.clone().detach()
        x = torch.nan_to_num(x, nan=0.0, posinf=5.0, neginf=-5.0)
        x[:, 0, :] = x[:, 0, :] * self.stats["lon_std"] + self.stats["lon_mean"]
        x[:, 1, :] = x[:, 1, :] * self.stats["lat_std"] + self.stats["lat_mean"]

        # 🔥 物理强约束：防止坐标飞出地球导致 Haversine 崩溃
        x[:, 0, :] = torch.clamp(x[:, 0, :], -180.0, 180.0)
        x[:, 1, :] = torch.clamp(x[:, 1, :], -90.0, 90.0)
        return x

    def _get_valid_length_mask(self, trajs):
        """
        🔥 [核心算法] 智能去噪 Mask
        自动检测轨迹末端的静止区域 (Padding 噪声)，返回有效长度的布尔掩码。
        适用于：Porto (短轨迹去噪) 和 Beijing/SF (尾部去噪)。
        """
        masks = []
        for t in trajs:
            # 计算每一步的位移
            diffs = np.linalg.norm(t[1:] - t[:-1], axis=1)

            is_moving = diffs > 1e-6

            # 从后往前扫描，找到第一个“动”的点，即为有效终点
            valid_len = len(t)
            for i in range(len(t) - 2, 0, -1):
                if is_moving[i]:
                    valid_len = i + 2
                    break

            # 构建 mask
            m = np.zeros(len(t), dtype=bool)
            m[:valid_len] = True
            masks.append(m)
        return masks

    def _get_len_raw_values(self, trajs, use_mask=True):
        """计算轨迹长度 (支持 Mask)"""
        vals = []
        is_beijing = 'beijing' in str(self.cfg.Data.DATASET).lower()

        # 获取掩码
        if use_mask:
            masks = self._get_valid_length_mask(trajs)
        else:
            masks = [np.ones(len(t), dtype=bool) for t in trajs]

        for i, t in enumerate(trajs):
            # 🔥 应用 Mask：只取有效点计算距离
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
                    # Porto 容错：跳跃过大 (>1km) 视为异常
                    if step <= 1.0: d += step
            vals.append(d)
        return vals

    def _get_len_dist(self, trajs, use_mask=True):
        """获取长度分布直方图"""
        raw_lens = self._get_len_raw_values(trajs, use_mask=use_mask)
        clipped = np.clip(raw_lens, 0.0, self.dynamic_limit)
        hist, _ = np.histogram(clipped, bins=np.linspace(0, self.dynamic_limit, 51))
        return hist.astype(float) / (hist.sum() + 1e-10)

    def _to_grid(self, trajs, bounds, grid_num=16):
        """映射到网格热力图 (通用)"""
        lon_min, lon_max = bounds["lon_bound"]
        lat_min, lat_max = bounds["lat_bound"]
        grid = np.zeros((grid_num, grid_num))

        if trajs.ndim == 3:
            all_pts = np.concatenate(trajs, axis=0)
        else:
            all_pts = trajs

        # 简单的 NaN 过滤
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

    # ================= 公开调用接口 =================
    @torch.no_grad()
    def evaluate(self, model, road_encoder, test_loader, sampling_func):
        model.eval()
        if road_encoder is not None:
            road_encoder.eval()

        gen_trajs = []

        # ==========================================
        # 1. 独立测速逻辑 (强制 BS=1)
        # ==========================================
        print("⏱️  [Evaluator] 正在执行单条推理测速 (BS=1)...")

        # 🔥 [兼容性] 安全获取单条数据
        dummy_batch = next(iter(test_loader))
        x0_single = dummy_batch[0][0:1].to(self.device)
        attr_single = dummy_batch[1][0:1].to(self.device)
        # 智能判断 batch 是否包含 road (DiffTraj=No, ControlTraj=Yes)
        road_single = dummy_batch[2][0:1].to(self.device) if len(dummy_batch) > 2 else None

        # 🔥 [兼容性] 自动构造 r_emb
        r_emb_single = None
        if road_encoder is not None and road_single is not None:
            # 模式 A: ControlTraj (有 Encoder + 有数据)
            r_in = self.road_processor.prepare_road_mae_input(road_single)
            r_emb_single = self.road_processor.extract_road_features(road_encoder, r_in, no_mask=True)
        elif hasattr(model, 'config') and hasattr(model.config, 'model'):
            # 模式 B: DiffTraj (无 Encoder, 需要占位符)
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

        # ==========================================
        # 2. 全量生成逻辑
        # ==========================================
        print("🚀 [Evaluator] 正在生成全量测试集 (Auto-Masking Enabled)...")

        for batch in tqdm(test_loader, desc="Generating"):
            # 🔥 [兼容性] 智能解包
            x0_real = batch[0].to(self.device)
            attr = batch[1].to(self.device)
            # 智能判断
            road = batch[2].to(self.device) if len(batch) > 2 else None

            # 🔥 [兼容性] 自动构造 r_emb
            r_emb = None
            if road_encoder is not None and road is not None:
                # 模式 A: 有路网约束
                r_in = self.road_processor.prepare_road_mae_input(road)
                r_emb = self.road_processor.extract_road_features(road_encoder, r_in, no_mask=True)
            elif hasattr(model, 'config') and hasattr(model.config, 'model'):
                # 模式 B: 无路网约束 (生成 DiffTraj 所需维度的零向量)
                r_emb = torch.zeros((len(x0_real), 1, model.config.model.ch)).to(self.device)

            x_noise = torch.randn_like(x0_real)

            # 生成
            x_gen = sampling_func(model, x_noise, r_emb, attr)

            # 反归一化
            gen_np = self._denormalize(x_gen).cpu().numpy().transpose(0, 2, 1)

            # ⚠️ 注意：这里不再进行 "Uniform Redistribute"，因为那会破坏尾部静止特征
            # 直接保存原始点，让后面的 Mask 逻辑去处理
            gen_trajs.append(gen_np)

        gen_all = np.concatenate(gen_trajs, axis=0)

        # ==========================================
        # 3. 计算指标 (核心)
        # ==========================================
        print("📊 [Evaluator] 计算指标中...")

        d_err = self._js_divergence(self.gt_density_hist, self._to_grid(gen_all, self.fixed_bounds))

        gen_trip_pts = np.stack([gen_all[:, 0, :], gen_all[:, -1, :]], axis=1)
        t_err = self._js_divergence(self.gt_trip_hist, self._to_grid(gen_trip_pts, self.fixed_bounds))

        # 🔥 Length Error: 重点！这里会自动调用 Mask 逻辑剔除噪声
        l_err = self._js_divergence(self.gt_len_hist, self._get_len_dist(gen_all, use_mask=True))

        # 诊断信息
        real_len_raw = self._get_len_raw_values(self.gt_trajs, use_mask=True)
        gen_len_raw = self._get_len_raw_values(gen_all, use_mask=True)

        print("\n" + "=" * 40)
        print("🕵️‍♂️ [DEBUG] 结果概览:")
        print(f"   ⏱️ Latency: {avg_latency:.2f} ms")
        print(f"   📏 Length Error: {l_err:.4f}")
        print(f"   🤖 Gen Avg Length: {np.mean(gen_len_raw):.4f} km (GT: {np.mean(real_len_raw):.4f})")
        print("=" * 40 + "\n")

        # 🤖 智能推断：根据经度均值判断城市
        # Evaluator 必定有 self.stats 用于反归一化，利用它！
        lon_mean = self.stats['lon_mean']

        if lon_mean > 100:  # 北京经度约 116
            dataset_name = "beijing"
        elif lon_mean < -100:  # 旧金山经度约 -122
            dataset_name = "sf"
        else:  # 波尔图经度约 -8
            dataset_name = "porto_HIDDEN_SIZE = 64"

        print(f"🌍 [Auto-Detect] 检测到数据集: {dataset_name} (Lon: {lon_mean:.1f})")

        # 保存文件
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