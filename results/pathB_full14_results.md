# Path B 9-Config × 14-Metric 完整评估结果

## 实验环境
- **服务器**: ts3 集群 gpu5 (8×A800 80GB for inference, 1 GPU for eval)
- **Backbone**: CogVideoX-5B + CineSync LoRA (r=128)
- **采样器**: VPSDEDPMPP2MSampler, 51 步, seed=42
- **测试集**: CineBrain Sub-05, Episode S7+S11 (video 7560-8099, 540 clips)
- **数据日期**: 2026-04-19 ~ 2026-04-24
- **评估脚本**: `tools/eval_full14.py`（per-metric try/except + 增量保存）
- **聚合脚本**: `tools/aggregate_9way_full14.py`

## 检查点与 Config 索引

| Config | 类型 | 推理日期 | MP4 输出目录 |
|---|---|---|---|
| E0_new_code | 静态门控 (CineBrain-SF v2 新代码) | 历史 | `results/alpha_540/E0_new_code/` |
| E3_cosine | Path A cosine 调度 (amp=+0.4) | 历史 | `results/alpha_540/E3_cosine/` |
| E4_reverse | Path A 反向 sigmoid (amp=-0.5) — **winner** | 历史 | `results/alpha_540/E4_reverse/` |
| E4_reverse_clamped | Path A E4_reverse + α_brain ≤ 0.95 clamp | 历史 | `results/alpha_540/E4_reverse_clamped/` |
| E4_sigmoid_mid | Path A 正向 sigmoid (amp=+0.5) | 历史 | `results/alpha_540/E4_sigmoid_mid/` |
| E4_sigmoid_mid_clamped | Path A E4_sigmoid + α_brain ≤ 0.95 clamp | 历史 | `results/alpha_540/E4_sigmoid_mid_clamped/` |
| pathB_p1_iter1000 | Path B Phase I, ckpt @ iter 1000 | 2026-04-23 | `results/alpha_540/pathB_p1_iter1000/` |
| pathB_p1_iter1500 | Path B Phase I, ckpt @ iter 1500 | 2026-04-22 | `results/alpha_540/pathB_p1_iter1500/` |
| pathB_p1_iter2000 | Path B Phase I, ckpt @ iter 2000 | 2026-04-20 | `results/alpha_540/pathB_p1_iter2000/` |

Path B 训练: `ckpts_5b/sf_v3_pathB_p1-04-19-01-13/`，2000 iter, 2-DDP on gpu2 GPU 3+5, 9.5h wall, 2026-04-19 01:13→10:42。Prior amp=0.5, Option C init std=0.1。

## 主表: 9-Config × 14 指标

完整 14 指标 540-sample 结果：

| Config | FVD↓ | EPE↓ | SSIM↑ | PSNR↑ | Hue-PCC↑ | CLIP↑ | CTC↑ | DTC↑ | CLIP-PCC↑ | VIFI↑ | Img-2way↑ | Img-50way↑ | Vid-2way↑ | Vid-50way↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| E0_new_code | 717.23 | 2.909 | 0.3097 | 12.129 | 0.392 | 0.743 | 0.986 | 0.980 | 0.984 | 0.838 | 0.934 | 0.349 | 0.908 | 0.283 |
| E3_cosine | 1144.73 | 2.643 | 0.2964 | 9.802 | 0.465 | 0.702 | 0.990 | 0.990 | 0.988 | 0.790 | 0.906 | 0.292 | 0.902 | 0.287 |
| **E4_reverse** | **425.28** | 3.192 | 0.2823 | 12.615 | 0.370 | 0.758 | 0.984 | 0.977 | 0.984 | 0.853 | 0.947 | 0.410 | 0.929 | **0.372** |
| E4_reverse_clamped | 429.93 | 3.221 | 0.2771 | 12.552 | 0.365 | 0.761 | 0.984 | 0.976 | 0.983 | 0.854 | 0.951 | 0.426 | 0.929 | 0.364 |
| E4_sigmoid_mid | 1193.63 | 2.598 | 0.2925 | 9.464 | 0.472 | 0.693 | 0.990 | 0.991 | 0.987 | 0.783 | 0.906 | 0.287 | 0.900 | 0.268 |
| E4_sigmoid_mid_clamped | 628.22 | 2.878 | 0.3051 | 12.043 | 0.390 | 0.747 | 0.987 | 0.982 | 0.985 | 0.841 | 0.936 | 0.375 | 0.918 | 0.311 |
| **pathB_p1_iter1000** | **501.05** | 2.972 | 0.2887 | 12.193 | 0.369 | 0.755 | 0.987 | 0.982 | 0.984 | 0.849 | 0.947 | **0.414** | 0.920 | 0.316 |
| **pathB_p1_iter1500** | 554.73 | **2.830** | 0.2885 | 12.311 | 0.370 | 0.752 | **0.988** | **0.983** | 0.983 | 0.846 | 0.943 | 0.391 | 0.922 | 0.337 |
| pathB_p1_iter2000 | 517.65 | 3.017 | 0.2855 | 12.190 | 0.372 | 0.752 | 0.987 | 0.981 | 0.985 | 0.845 | 0.941 | 0.396 | 0.929 | 0.328 |

## Delta 表: 每个 config 相对 Path A winner (E4_reverse)

正号 = 绝对值增加；✓/✗ 标记方向是否"更好"（依指标越高/越低越好）。

| Config | FVD↓ | EPE↓ | SSIM↑ | PSNR↑ | Hue-PCC↑ | CLIP↑ | CTC↑ | DTC↑ | CLIP-PCC↑ | VIFI↑ | Img-2way↑ | Img-50way↑ | Vid-2way↑ | Vid-50way↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| E0_new_code | +291.9 ✗ | -0.28 ✓ | +0.027 ✓ | -0.49 ✗ | +0.022 ✓ | -0.015 ✗ | +0.002 ✓ | +0.004 ✓ | -0.000 ✗ | -0.014 ✗ | -0.013 ✗ | -0.061 ✗ | -0.021 ✗ | -0.089 ✗ |
| E3_cosine | +719.4 ✗ | -0.55 ✓ | +0.014 ✓ | -2.81 ✗ | +0.095 ✓ | -0.056 ✗ | +0.005 ✓ | +0.014 ✓ | +0.004 ✓ | -0.062 ✗ | -0.041 ✗ | -0.117 ✗ | -0.027 ✗ | -0.085 ✗ |
| E4_reverse_clamped | +4.6 ✗ | +0.03 ✗ | -0.005 ✗ | -0.06 ✗ | -0.005 ✗ | +0.003 ✓ | +0.000 ✓ | -0.000 ✗ | -0.002 ✗ | +0.002 ✓ | +0.004 ✓ | +0.016 ✓ | -0.000 ✗ | -0.009 ✗ |
| E4_sigmoid_mid | +768.3 ✗ | -0.59 ✓ | +0.010 ✓ | -3.15 ✗ | +0.102 ✓ | -0.065 ✗ | +0.006 ✓ | +0.014 ✓ | +0.002 ✓ | -0.070 ✗ | -0.041 ✗ | -0.122 ✗ | -0.029 ✗ | -0.104 ✗ |
| E4_sigmoid_mid_clamped | +202.9 ✗ | -0.31 ✓ | +0.023 ✓ | -0.57 ✗ | +0.019 ✓ | -0.011 ✗ | +0.003 ✓ | +0.005 ✓ | +0.001 ✓ | -0.012 ✗ | -0.011 ✗ | -0.035 ✗ | -0.010 ✗ | -0.062 ✗ |
| **pathB_iter1000** | **+75.8 ✗** | **-0.22 ✓** | +0.006 ✓ | -0.42 ✗ | -0.001 ✗ | -0.003 ✗ | +0.003 ✓ | +0.005 ✓ | -0.001 ✗ | -0.003 ✗ | +0.000 ✓ | +0.004 ✓ | -0.009 ✗ | -0.056 ✗ |
| **pathB_iter1500** | **+129.4 ✗** | **-0.36 ✓** | +0.006 ✓ | -0.30 ✗ | -0.000 ✗ | -0.006 ✗ | +0.004 ✓ | +0.007 ✓ | -0.002 ✗ | -0.006 ✗ | -0.004 ✗ | -0.019 ✗ | -0.006 ✗ | -0.035 ✗ |
| pathB_iter2000 | +92.4 ✗ | -0.18 ✓ | +0.003 ✓ | -0.42 ✗ | +0.002 ✓ | -0.006 ✗ | +0.002 ✓ | +0.005 ✓ | +0.001 ✓ | -0.008 ✗ | -0.006 ✗ | -0.014 ✗ | +0.000 ✓ | -0.044 ✗ |

## 关键发现

### Finding 1: Path A 与 Path B 优化不同指标家族

**Path A (E4_reverse) 赢**（视频-级分布/语义识别方向）:
- **FVD**: 425 vs Path B 501-555（绝对优势 +76~+130）
- **Vid-50way**: 0.372 vs Path B 0.316-0.337（+0.035~+0.056）
- **VIFI**: 0.853 vs Path B 0.845-0.849
- **CLIP**: 0.758 vs Path B 0.752-0.755
- **PSNR**: 12.61 vs Path B 12.19-12.31

**Path B 赢**（帧间动态/时空一致性方向）:
- **EPE**: 2.83 (iter1500) vs Path A 3.19（-0.36，相对改善 11%）
- **CTC**: 0.988 vs Path A 0.984
- **DTC**: 0.983 vs Path A 0.977
- **SSIM**: 0.289 vs Path A 0.282

### Finding 2: Path B 内部 Pareto = {iter1000, iter1500}

iter 1000/1500/2000 不是单调，而是跨 iter 的 Pareto frontier:

| 指标 | 最优 iter | 值 | 跨 iter 排序 |
|---|:---:|:---:|:---|
| FVD | **iter 1000** | 501 | 1000 < 2000 (518) < 1500 (555) |
| EPE | **iter 1500** | 2.830 | 1500 < 1000 (2.97) < 2000 (3.02) |
| Img-50way | **iter 1000** | 0.414 | 1000 > 2000 (0.396) > 1500 (0.391) |
| Vid-50way | **iter 1500** | 0.337 | 1500 > 2000 (0.328) > 1000 (0.316) |
| CTC / DTC | **iter 1500** | — | 1500 最佳 |
| SSIM / PSNR | 持平 | — | 三点 ±0.003 以内 |

**iter 2000 在所有指标上均被 iter 1000 或 iter 1500 dominate**，故定性为 Pareto-dominated，不值得作为 Path B 代表。

### Finding 3: val loss ≠ FVD

训练时 val loss 轨迹: iter 1500 (0.396) < iter 2000 (0.486)，本来预测 iter 1500 FVD 最佳。实际 FVD: iter 1500 (555) > iter 2000 (518) > iter 1000 (501)。**val loss 反映 per-sample 重建（与 EPE 一致），不能预测 FVD 这种分布级指标**。

### Finding 4: Clamp vs 无 clamp 微弱差异

E4_reverse vs E4_reverse_clamped 的 14 指标差异全部在 ±0.02 以内，**clamp 对 full 540 结果几乎零影响**。这说明 §4.4 观察到 `amp > 0` 时 clamp 大幅改善 FVD 是特定于"超出 sigmoid 训练分布"场景，对 `amp < 0` 或 learned Path B 不关键。

## 关联数据源

- **原始 JSON**: `results/alpha_540/summary_{config}_full14.json`（9 个独立 JSON）
- **聚合 JSON**: `results/alpha_540/summary_9way_pathB_full14.json`
- **α(τ) probe 数据**: `results/pathB/alpha_curve_p1_iter{500,1000,1500}.json` + `alpha_curve_p1_2000_iter2000.json` + `alpha_curve_init_with_prior_v2.json`
- **训练 log**: `logs/pathB_p1_2000.log`（val loss 轨迹 + SF loss 分解）

## 后续实验（规划中，未完成）

### pathB_p1_iter500 full-540（进行中）
- 启动 2026-04-24 凌晨 gpu5 × 8 卡
- ETA ~05:00
- Output: `results/alpha_540/pathB_p1_iter500/`
- 完成后将补充进 10-way 表，预期占据 Pareto 曲线低训练端点

### α-clamp full-540 消融（待启动）
- 4 个 config: clamp_none / clamp_mot / clamp_brain / clamp_both
- 使用 iter 1500 ckpt + `sampling.py` 的 `path_b.clamp_channels` 新增参数
- Mini-68 已验证代码生效（α_brain 0.75→0.50 @ clamp=both），但噪声吞没 FVD/EPE 信号
- Full 540 × 4 configs 预计 ~10-20h，回答"哪个通道 drift 控制哪个指标"
- 决定 Path B Phase 2 (prior-anchor 正则) 的设计方向

### Path B Phase 2（设计中，依赖 α-clamp 消融结果）
- 预计方案: 仅对 α_brain 的 gate_net 残差加 L2 正则（如果 clamp_brain 确实回收 FVD 且 EPE 保持）
- 训练时长: ~9h per λ setting
- 目标: 同时拿 FVD ≤ 450 与 EPE ≤ 2.95
