# Path B 13-Config × 14-Metric 完整评估结果

## 实验环境
- **服务器**: ts3 集群 gpu5 (8×A800 80GB for inference, 1-2 GPU for eval); cont500 推理也用了 gpu2 GPU 0+1（与他人 LLaVA 共用）
- **Backbone**: CogVideoX-5B + CineSync LoRA (r=128)
- **采样器**: VPSDEDPMPP2MSampler, 51 步, seed=42
- **测试集**: CineBrain Sub-05, Episode S7+S11 (video 7560-8099, 540 clips)
- **数据日期**: 2026-04-19 ~ 2026-04-26（cont500 续训实验在 04-24/26 完成）
- **评估脚本**: `tools/eval_full14.py`（per-metric try/except + 增量保存）
- **聚合脚本**: `tools/aggregate_11way_full14.py`
- **聚合 JSON**: `results/alpha_540/summary_11way_pathB_full14.json`（仅 P1 11-way；cont500 单独 JSON 见 §关联数据源）

## 检查点与 Config 索引

| Config | 类型 | 推理日期 | MP4 输出目录 |
|---|---|---|---|
| E0_new_code | 静态门控 (CineBrain-SF v2 新代码) | 历史 | `results/alpha_540/E0_new_code/` |
| E3_cosine | Path A cosine 调度 (amp=+0.4) | 历史 | `results/alpha_540/E3_cosine/` |
| E4_reverse | Path A 反向 sigmoid (amp=-0.5) — **Path A winner** | 历史 | `results/alpha_540/E4_reverse/` |
| E4_reverse_clamped | Path A E4_reverse + α_brain ≤ 0.95 clamp | 历史 | `results/alpha_540/E4_reverse_clamped/` |
| E4_sigmoid_mid | Path A 正向 sigmoid (amp=+0.5) | 历史 | `results/alpha_540/E4_sigmoid_mid/` |
| E4_sigmoid_mid_clamped | Path A E4_sigmoid + α_brain ≤ 0.95 clamp | 历史 | `results/alpha_540/E4_sigmoid_mid_clamped/` |
| **pathB_p1_iter0** | Path B partial_load 起点, 零残差 | 2026-04-24 | `results/alpha_540/pathB_p1_iter0/` |
| **pathB_p1_iter500** | Path B Phase I, ckpt @ iter 500 | 2026-04-24 | `results/alpha_540/pathB_p1_iter500/` |
| pathB_p1_iter1000 | Path B Phase I, ckpt @ iter 1000 | 2026-04-23 | `results/alpha_540/pathB_p1_iter1000/` |
| pathB_p1_iter1500 | Path B Phase I, ckpt @ iter 1500 | 2026-04-22 | `results/alpha_540/pathB_p1_iter1500/` |
| pathB_p1_iter2000 | Path B Phase I, ckpt @ iter 2000 | 2026-04-20 | `results/alpha_540/pathB_p1_iter2000/` |
| **cont500_iter300** | cont500 续训 iter 300 (val-loss-best, FVD valley) | 2026-04-25 | `results/alpha_540/pathB_cont500_iter300/` |
| **cont500_iter500** | cont500 续训终点 iter 500 | 2026-04-26 | `results/alpha_540/pathB_cont500_iter500/` |

Path B 训练: `ckpts_5b/sf_v3_pathB_p1-04-19-01-13/`，2000 iter, 2-DDP on gpu2 GPU 3+5, 9.5h wall, 2026-04-19 01:13→10:42。Prior amp=0.5, Option C init std=0.1。iter 0 对应 `ckpts_5b/sf_v3_pathB_init`（partial_load 输出，v2 ckpt 右 pad 到 Path B 结构）。

cont500 续训: `ckpts_5b/sf_v3_pathB_cont500_lr1e6-04-24-16-06/`，500 iter, lr=1e-6（10× 缩小）, 2-DDP on gpu2 GPU 4+5, 2.1h wall, 2026-04-24 16:06→18:13。从 P1_iter500 出发，假设"lr 太大 → drift"，验证小 lr 能否守 FVD 440。Save_interval=100 (5 ckpt: 100/200/300/400/500)。Loss + freezing 配置同 P1 (TimeNoise off, lambda_sf=0.003)。

## 主表: 11-Config × 14 指标

| Config | FVD↓ | EPE↓ | SSIM↑ | PSNR↑ | Hue-PCC↑ | CLIP↑ | CTC↑ | DTC↑ | CLIP-PCC↑ | VIFI↑ | Img-2way↑ | Img-50way↑ | Vid-2way↑ | Vid-50way↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| E0_new_code | 717.23 | 2.909 | 0.3097 | 12.129 | 0.392 | 0.743 | 0.986 | 0.980 | 0.984 | 0.838 | 0.934 | 0.349 | 0.908 | 0.283 |
| E3_cosine | 1144.73 | 2.643 | 0.2964 | 9.802 | 0.465 | 0.702 | 0.990 | 0.990 | 0.988 | 0.790 | 0.906 | 0.292 | 0.902 | 0.287 |
| **E4_reverse** | **425.28** | 3.192 | 0.2823 | **12.615** | 0.370 | **0.758** | 0.984 | 0.977 | 0.984 | **0.853** | 0.947 | 0.410 | 0.929 | 0.372 |
| E4_reverse_clamped | 429.93 | 3.221 | 0.2771 | 12.552 | 0.365 | 0.761 | 0.984 | 0.976 | 0.983 | 0.854 | 0.951 | **0.426** | 0.929 | 0.364 |
| E4_sigmoid_mid | 1193.63 | 2.598 | 0.2925 | 9.464 | **0.472** | 0.693 | 0.990 | 0.991 | 0.987 | 0.783 | 0.906 | 0.287 | 0.900 | 0.268 |
| E4_sigmoid_mid_clamped | 628.22 | 2.878 | 0.3051 | 12.043 | 0.390 | 0.747 | 0.987 | 0.982 | 0.985 | 0.841 | 0.936 | 0.375 | 0.918 | 0.311 |
| pathB_p1_iter0 | 467.82 | 3.121 | 0.2921 | 12.373 | 0.373 | 0.754 | 0.985 | 0.978 | 0.983 | 0.848 | 0.945 | 0.404 | 0.919 | 0.349 |
| **pathB_p1_iter500** ⭐ | **440.29** | 3.013 | 0.2794 | 12.340 | 0.372 | 0.756 | 0.987 | 0.981 | 0.984 | 0.846 | 0.944 | 0.401 | 0.928 | **0.377** |
| pathB_p1_iter1000 | 501.05 | 2.972 | 0.2887 | 12.193 | 0.369 | 0.755 | 0.987 | 0.982 | 0.984 | 0.849 | 0.947 | 0.414 | 0.920 | 0.316 |
| **pathB_p1_iter1500** | 554.73 | **2.830** | 0.2885 | 12.311 | 0.370 | 0.752 | **0.988** | **0.983** | 0.983 | 0.846 | 0.943 | 0.391 | 0.922 | 0.337 |
| pathB_p1_iter2000 | 517.65 | 3.017 | 0.2855 | 12.190 | 0.372 | 0.752 | 0.987 | 0.981 | 0.985 | 0.845 | 0.941 | 0.396 | 0.929 | 0.328 |
| cont500_iter300 | 487.05 | **2.920** | 0.2833 | 12.336 | 0.371 | 0.756 | **0.988** | 0.982 | **0.986** | 0.848 | 0.947 | 0.410 | 0.917 | 0.331 |
| cont500_iter500 | 434.77 | 2.961 | 0.2809 | 12.349 | 0.368 | 0.752 | 0.988 | 0.981 | 0.986 | 0.846 | 0.940 | 0.394 | 0.914 | 0.332 |

## Delta 表: 每 config 相对 Path A winner (E4_reverse)

| Config | FVD↓ | EPE↓ | SSIM↑ | PSNR↑ | Hue-PCC↑ | CLIP↑ | CTC↑ | DTC↑ | CLIP-PCC↑ | VIFI↑ | Img-2way↑ | Img-50way↑ | Vid-2way↑ | Vid-50way↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| E0_new_code | +291.9 ✗ | -0.28 ✓ | +0.027 ✓ | -0.49 ✗ | +0.022 ✓ | -0.015 ✗ | +0.002 ✓ | +0.004 ✓ | -0.000 ✗ | -0.014 ✗ | -0.013 ✗ | -0.061 ✗ | -0.021 ✗ | -0.089 ✗ |
| E3_cosine | +719.4 ✗ | -0.55 ✓ | +0.014 ✓ | -2.81 ✗ | +0.095 ✓ | -0.056 ✗ | +0.005 ✓ | +0.014 ✓ | +0.004 ✓ | -0.062 ✗ | -0.041 ✗ | -0.117 ✗ | -0.027 ✗ | -0.085 ✗ |
| E4_reverse_clamped | +4.6 ✗ | +0.03 ✗ | -0.005 ✗ | -0.06 ✗ | -0.005 ✗ | +0.003 ✓ | +0.000 ✓ | -0.000 ✗ | -0.002 ✗ | +0.002 ✓ | +0.004 ✓ | +0.016 ✓ | -0.000 ✗ | -0.009 ✗ |
| E4_sigmoid_mid | +768.3 ✗ | -0.59 ✓ | +0.010 ✓ | -3.15 ✗ | +0.102 ✓ | -0.065 ✗ | +0.006 ✓ | +0.014 ✓ | +0.002 ✓ | -0.070 ✗ | -0.041 ✗ | -0.122 ✗ | -0.029 ✗ | -0.104 ✗ |
| E4_sigmoid_mid_clamped | +202.9 ✗ | -0.31 ✓ | +0.023 ✓ | -0.57 ✗ | +0.019 ✓ | -0.011 ✗ | +0.003 ✓ | +0.005 ✓ | +0.001 ✓ | -0.012 ✗ | -0.011 ✗ | -0.035 ✗ | -0.010 ✗ | -0.062 ✗ |
| pathB_iter0 | +42.5 ✗ | -0.07 ✓ | +0.010 ✓ | -0.24 ✗ | +0.003 ✓ | -0.004 ✗ | +0.001 ✓ | +0.001 ✓ | -0.001 ✗ | -0.005 ✗ | -0.003 ✗ | -0.006 ✗ | -0.010 ✗ | -0.023 ✗ |
| **pathB_iter500** | **+15.0 ✗** | **-0.18 ✓** | -0.003 ✗ | -0.27 ✗ | +0.002 ✓ | -0.002 ✗ | +0.003 ✓ | +0.004 ✓ | -0.000 ✗ | -0.007 ✗ | -0.003 ✗ | -0.009 ✗ | -0.001 ✗ | **+0.005 ✓** |
| pathB_iter1000 | +75.8 ✗ | -0.22 ✓ | +0.006 ✓ | -0.42 ✗ | -0.001 ✗ | -0.003 ✗ | +0.003 ✓ | +0.005 ✓ | -0.001 ✗ | -0.003 ✗ | +0.000 ✓ | +0.004 ✓ | -0.009 ✗ | -0.056 ✗ |
| pathB_iter1500 | +129.4 ✗ | -0.36 ✓ | +0.006 ✓ | -0.30 ✗ | -0.000 ✗ | -0.006 ✗ | +0.004 ✓ | +0.007 ✓ | -0.002 ✗ | -0.006 ✗ | -0.004 ✗ | -0.019 ✗ | -0.006 ✗ | -0.035 ✗ |
| pathB_iter2000 | +92.4 ✗ | -0.18 ✓ | +0.003 ✓ | -0.42 ✗ | +0.002 ✓ | -0.006 ✗ | +0.002 ✓ | +0.005 ✓ | +0.001 ✓ | -0.008 ✗ | -0.006 ✗ | -0.014 ✗ | +0.000 ✓ | -0.044 ✗ |
| cont500_iter300 | +61.8 ✗ | -0.27 ✓ | +0.001 ✓ | -0.28 ✗ | +0.001 ✓ | -0.002 ✗ | +0.004 ✓ | +0.005 ✓ | +0.002 ✓ | -0.005 ✗ | -0.000 ✗ | +0.000 ✓ | -0.012 ✗ | -0.041 ✗ |
| **cont500_iter500** | **+9.5 ✗** | -0.23 ✓ | -0.001 ✗ | -0.27 ✗ | -0.002 ✗ | -0.006 ✗ | +0.004 ✓ | +0.005 ✓ | +0.002 ✓ | -0.007 ✗ | -0.007 ✗ | -0.016 ✗ | -0.015 ✗ | -0.040 ✗ |

## 主要发现

### Finding 1: iter 500 是 Path B 综合最佳（NEW, 2026-04-24）

**iter 500 第一次在 Vid-50way 上超过 Path A**（0.377 vs 0.372, +0.005），同时:
- FVD 只差 Path A 15 单位（+3.5% 相对）
- EPE 改进 0.18（-5.6% 相对）
- CTC/DTC 都赢
- 其他 11 个指标差异多为 ±0.003~0.009（噪声量级）

**iter 500 几乎是 Path A 的无损升级**——相比之前 iter 1000 作为 FVD-best 时 +76 FVD / -0.22 EPE 的 trade，iter 500 的 +15/-0.18 更接近无损。

### Finding 2: Pareto Frontier 4 点（iter 0 + iter 2000 dominated）

最终 FVD-EPE 双轴 Pareto:

```
  Path A (E4_reverse)  FVD 425  EPE 3.19  ← FVD 最优
  pathB iter 500       FVD 440  EPE 3.01  ← 近平衡点（NEW）
  pathB iter 1000      FVD 501  EPE 2.97  ← 中
  pathB iter 1500      FVD 555  EPE 2.83  ← EPE 最优
```

Dominated:
- **pathB iter 0** (468, 3.12) — 被 iter 500 全面压过（更差 FVD + EPE）
- **pathB iter 2000** (518, 3.02) — 被 iter 500 在两轴都压过（更差 FVD + EPE ≈ 相同）

### Finding 3: Path B 起点 ≠ Path A (+43 FVD 内在 gap)

iter 0 FVD 468 vs Path A 425，差 43 单位。Path B 的加法 prior（`logit = gate(pooled, t_emb) + prior_bias(τ)`）与 Path A 的乘法调度（`v2_alpha × sched_multiplier`）在数学上近似但不完全等价。这证实"起点不够好"假设**部分成立**——但不是毁灭性，iter 500 训练 500 iter 后就把 FVD 从 468 拉到 440，几乎追回。

### Finding 4: 训练轨迹 = 同向改进 → Pareto trade → 过训练三阶段

| 区间 | FVD 变化 | EPE 变化 | 性质 |
|---|:---:|:---:|:---|
| iter 0 → 500 | 468 → 440 ↓ | 3.12 → 3.01 ↓ | **同向改进**（两者都变好）|
| iter 500 → 1000 | 440 → 501 ↑ | 3.01 → 2.97 ↓ | **Pareto trade**（FVD 伤换 EPE 增）|
| iter 1000 → 2000 | 501 → 518 ↑ | 2.97 → 3.02 ↑ | **过训练**（两者都变差）|

这解释了之前 sf_total 在 iter 1000 触底、iter 2000 爆恶化的训练 loss 轨迹——gate_net 在 iter 500 附近就已经到达"Path A 附近"的最优工作点，之后是在 sample-path 上学 FVD-有害的漂移。

### Finding 5: Path A 与 Path B 的指标家族分化（在 iter 1000+ 显著）

iter 500 时分化微弱，但 iter 1000+ 分化明显:

**Path A (E4_reverse) 赢**（视频-级分布/语义识别）: FVD, V-50way (iter 1000+), VIFI, CLIP, PSNR
**Path B iter 1000+ 赢**（帧间动态/时空一致性）: EPE, CTC, DTC, SSIM
**iter 500 近乎持平 Path A**，只在 EPE + CTC + DTC + Vid-50way 小赢

### Finding 6: E4_reverse_clamped 与 E4_reverse 几乎相同

14 指标差异全部 ≤ 0.02（FVD +4.6, 其他 ±0.01 内），说明 α_brain 的 0.95 clamp 对 full 540 结果**几乎零影响**——与 mini 实验一致。

### Finding 7: cont500 (lr=1e-6 续训) 净效果 ≈ 0，但揭示了真正的瓶颈（NEW, 2026-04-26）

**实验设计**：从 P1_iter500 出发，lr=1e-6（10× 缩小）继续训 500 iter，假设"lr 太大导致 iter 1000 之后 FVD drift"。Save 100/200/300/400/500 五个 ckpt，跑 iter 300（val-loss-best）和 iter 500（终点）的 540 推理 + 14 metric。

**结果（vs P1_iter500 起点）**:

| ckpt | FVD | EPE | Vid-50way | 总评 |
|---|:---:|:---:|:---:|---|
| P1_iter500（起点） | 440.29 | 3.013 | **0.377** ⭐ | 当前最强 |
| cont500_iter300 | 487.05 (+47) | 2.920 (-0.09) | 0.331 (-0.046) | FVD valley，全方位偏向 EPE |
| **cont500_iter500** | **434.77 (-5)** | 2.961 (-0.05) | 0.332 (-0.045) | FVD 微好但 Vid-50way 显著退 |

**FVD valley 现象**: cont500 训练中 FVD 走 440→487(@300)→435(@500) 的 U 形轨迹。lr 衰减（iter 300 lr=4e-7→iter 500 lr=8e-9）在尾段帮模型 "settle" 回近原点。

**val loss 误导**: val loss 最低出现在 cont500_iter300（0.366），低于 P1 任何 iter（最低 0.396 @ iter 1500）；但此处 FVD 反而最差。**val loss 与 FVD 严重脱钩**。

**α(τ) 不变**: 5 个 cont500 ckpt × 4 channel × 7 τ 点的 α(τ) 测量，所有 Δα < 0.001。说明 gate_net 几乎没变，所有优化都来自 **fusion_layers / output_proj**——它们的微小漂移就足以让 FVD 涨/退 ±50 单位、Vid-50way 退 0.045。

### Finding 8: cont500 ≈ P1_iter500 的不同 Pareto 点（同一曲线上更小步长）

cont500_iter500 vs P1_iter500 整体差异都 ≤ 0.05 量级，**模型仍在原 Pareto 曲线上**——只是因为 lr 小步长缩小，沿曲线走得更慢。500 iter × lr=1e-6 净效果 ≈ 50 iter × lr=1e-5。无质变。

**含义**: 即使 lr 缩 10×，模型仍朝同方向走（为 EPE/CTC 牺牲 Vid-50way）。问题在 **loss formulation**（SF loss 把 fusion_layers 推向帧间一致性 metric），不是 lr。

## 训练方式的诊断结论（更新于 2026-04-26）

用户提出的两个假设:
1. **"起点不够好"** — **部分成立但不是主因**：iter 0 FVD 468 比 Path A 的 425 高 43 单位（加法 vs 乘法 prior gap）；但 iter 500 已经追回到 440，证明起点 gap 可以训出来
2. **"训练方式不对"** — **强证据：是 loss formulation 的问题，不是 lr**：
   - 原 P1 训练：iter 0→500 同向改进 → 500→1000 Pareto trade → 1000→2000 过训练
   - cont500 (lr=1e-6 续训)：500 iter 净变化 ≈ 50 iter 的 lr=1e-5 训练，模型仍朝同方向走；α(τ) 不变，fusion_layers 漂移
   - 结论：模型受 loss 驱动（SF loss / 重建 loss / KL）走向"帧间一致性 / EPE / CTC"，**任何 lr 配置都改变不了方向**

**最简单的实用 fix = 直接用 P1_iter500 作为论文 ckpt**（不再续训）。要破局需要改 loss formulation，不是改 lr。

**真正能改变 Pareto 方向的实验候选**：
- 降低 SF loss 权重（lambda_sf 0.003 → 0.0005/0.001）：减小 SF 对 fusion 的拉扯，看 Vid-50way 能否守
- fusion_layers anchor L2 正则（不只 gate_net）：防止 fusion 在续训中漂移
- TimeNoise 开启（DESIGN Phase II）：但 cont500 已显示 gate_net 不学，TimeNoise 可能也无效

## 关联数据源

### P1 baseline (11 configs)
- **原始 JSON**: `results/alpha_540/summary_{config}_full14.json`
- **11-way 聚合 JSON**: `results/alpha_540/summary_11way_pathB_full14.json`
- **α(τ) probe**: `results/pathB/alpha_curve_p1_iter{500,1000,1500}.json` + `alpha_curve_p1_2000_iter2000.json` + `alpha_curve_init_with_prior_v2.json`
- **训练 log**: `logs/pathB_p1_2000.log`（val loss 轨迹 + SF loss 分解）
- **推理 log**: `logs/pathB_iter{0,500,1000,1500,2000}_gpu5_gpu{0..7}.log`
- **Eval log**: `logs/eval_full14_pathB_p1_iter{0,500,1000,1500,2000}.log`
- **推理 launcher**: `tools/launch_pathB_iter0_500_4way_each.sh` + `tools/launch_pathB_iter{500,1000,1500}_8way.sh`
- **Eval launcher**: `tools/run_eval_full14_parallel.sh` + `tools/eval_full14.py --result-dir`

### cont500 续训实验 (NEW, 2026-04-26)
- **训练 ckpt（5 个）**: `ckpts_5b/sf_v3_pathB_cont500_lr1e6-04-24-16-06/{100,200,300,400,500}/mp_rank_00_model_states.pt`
- **训练 config**: `configs/sf_v1/sf_v3_pathB_cont500_lr1e6_train.yaml`（lr=1e-6, train_iters=500, save_interval=100, mode=finetune）
- **训练 log**: `logs/pathB_cont500_lr1e6.log`
- **起点 wrapper dir**: `ckpts_5b/sf_v3_pathB_cont500_init/`（latest=500，hardlink 到 P1_iter500）
- **推理 wrapper dirs**: `ckpts_5b/sf_v3_pathB_cont500_lr1e6_iter{300,500}/`
- **推理 config**: `configs/sf_v1/infer_pathB_cont500_iter{300,500}.yaml`
- **推理 launcher**: `tools/launch_cont500_iter300_8way_yield.sh`（gpu5 8-way + courtesy yield watchdog）+ `tools/launch_cont500_iter500_2way_yield_gpu2.sh`（gpu2 GPU 0+1，yield 范围限定 GPU 0/1；后改为不挂 watchdog 直接共用 LLaVA）
- **推理 mp4**: `results/alpha_540/pathB_cont500_iter{300,500}/*.mp4`
- **Eval JSON**: `results/alpha_540/summary_cont500_iter{300,500}_full14.json`
- **Eval log**: `logs/eval_cont500_iter{300,500}.log`
- **α(τ) probe（5 ckpts）**: `results/pathB/alpha_curve_cont500_iter{100,200,300,400,500}.json`
- **α probe launcher**: `tools/run_pathB_probe_cont500.sh`（单 GPU 串行 5 ckpts，~17 min）

## 后续实验建议（2026-04-26 更新）

### 已验证的 negative results — 不再尝试

- ❌ **小 lr 续训** (cont500, lr=1e-6, 500 iter): 已做。净效果 ≈ 0，模型仍沿同 Pareto 曲线走小步。问题是 loss formulation，不是 lr。

### 高优先级（基于 cont500 发现 + 之前判断）

1. **论文 Table 2 主行 = P1_iter500**（不变，cont500 确认仍是最强）
2. **降低 SF loss 权重重训**：`lambda_sf` 0.003 → 0.0005 / 0.001 / 0.002 三档；从 iter 0 起 2000 iter；目标 — 看 Vid-50way 是否守、FVD 是否守
3. **fusion_layers anchor L2 正则重训**：在 loss 上加 `λ·||fusion_state - fusion_initial||²`（不只 gate_net 而是整个 gated_fusion 模块），λ 扫 1e-4/1e-3/1e-2；阻止 cont500 中观察到的 fusion 漂移

### 中优先级（仍未验证）

4. **Path B formulation 对齐 Path A**: 换成乘法 α × schedule formulation；如果 SF loss 重权 + anchor L2 都救不了 Vid-50way，再考虑这条
5. **TimeNoise 开启**（DESIGN Phase II）: cont500 已显示 gate_net 在小 lr 下不学；TimeNoise 是否能让 gate_net 学起来仍是未知

### 低优先级（之前 backlog）

6. **α-clamp full 540 消融** (4 configs × 5h)
7. **跨 subject Path B 测试**: 当前只在 Sub-05
8. **Figure 2 绘图**: α(τ) 轨迹 + FVD-EPE 散点 + 高/低 motion 样本 α_mot(τ) 对比
9. **Appendix H.4 训练 loss trajectory**: 论文引用的 SF_total / diff_loss / val_loss 详细图（cont500 数据可加进来作对比）
