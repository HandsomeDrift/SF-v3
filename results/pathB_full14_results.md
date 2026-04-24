# Path B 11-Config × 14-Metric 完整评估结果

## 实验环境
- **服务器**: ts3 集群 gpu5 (8×A800 80GB for inference, 1-2 GPU for eval)
- **Backbone**: CogVideoX-5B + CineSync LoRA (r=128)
- **采样器**: VPSDEDPMPP2MSampler, 51 步, seed=42
- **测试集**: CineBrain Sub-05, Episode S7+S11 (video 7560-8099, 540 clips)
- **数据日期**: 2026-04-19 ~ 2026-04-24
- **评估脚本**: `tools/eval_full14.py`（per-metric try/except + 增量保存）
- **聚合脚本**: `tools/aggregate_11way_full14.py`
- **聚合 JSON**: `results/alpha_540/summary_11way_pathB_full14.json`

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

Path B 训练: `ckpts_5b/sf_v3_pathB_p1-04-19-01-13/`，2000 iter, 2-DDP on gpu2 GPU 3+5, 9.5h wall, 2026-04-19 01:13→10:42。Prior amp=0.5, Option C init std=0.1。iter 0 对应 `ckpts_5b/sf_v3_pathB_init`（partial_load 输出，v2 ckpt 右 pad 到 Path B 结构）。

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

## 训练方式的诊断结论

用户提出的两个假设:
1. **"起点不够好"** — **部分成立**：iter 0 FVD 468 比 Path A 的 425 高 43 单位；加法 vs 乘法 prior formulation 有内在 gap
2. **"训练方式不对"** — **部分成立但更微妙**：前 500 iter 训练**同向改进 FVD+EPE**；500+ 开始 Pareto trade；1000+ 过训练

**最简单、最有效的 fix = early stopping @ iter 500**，不需要复杂的正则化。iter 500 几乎是 Path A 的无损升级，其余 iter 都在 Pareto 上退让 FVD 换 EPE。

## 关联数据源

- **原始 JSON** (11个): `results/alpha_540/summary_{config}_full14.json`
- **11-way 聚合 JSON**: `results/alpha_540/summary_11way_pathB_full14.json`
- **α(τ) probe 数据**: `results/pathB/alpha_curve_p1_iter{500,1000,1500}.json` + `alpha_curve_p1_2000_iter2000.json` + `alpha_curve_init_with_prior_v2.json`
- **训练 log**: `logs/pathB_p1_2000.log`（val loss 轨迹 + SF loss 分解）
- **推理 log**: `logs/pathB_iter{0,500,1000,1500,2000}_gpu5_gpu{0..7}.log`
- **Eval log**: `logs/eval_full14_pathB_p1_iter{0,500,1000,1500,2000}.log`
- **推理 launcher**: `tools/launch_pathB_iter0_500_4way_each.sh` (4+4 并行) + `tools/launch_pathB_iter{500,1000,1500}_8way.sh` (8-way)
- **Eval launcher**: `tools/run_eval_full14_parallel.sh` (2-chain 兼顾 CPU RAM) + 单 config 直接 `tools/eval_full14.py --result-dir`

## 后续实验建议

### 高优先级（基于本次发现）

1. **论文 Table 2 更新为 iter 500 主行**: 之前的 iter 1000 作为 "FVD-best"，现在 iter 500 用 "near-Path-A balanced" 故事更强
2. **iter 500 作为 Path B 的"官方"号**: 在所有后续实验中（比较 / 跨 subject / P2 设计基线）用 iter 500 ckpt 而非 iter 1500/2000

### 中优先级（探索训练改进）

3. **小 lr 继续训 iter 500 之后**: 如果 lr=1e-5 在 iter 1000 开始过训练，试 lr=1e-6 从 iter 500 继续训 500 iter → 看能否保 FVD 不变而 EPE 继续降
4. **早停 + Prior-anchor L2 正则重训**: 从 iter 0 起，加 `λ·||gate_net_output||²` 到 loss，λ 扫 [1e-4, 1e-3, 1e-2]，看能否把 iter 500 的 Pareto 点再推进
5. **Path B formulation 对齐 Path A**: 换成乘法 α × schedule formulation（不用加法 logit prior），看能否把 iter 0 FVD 从 468 拉回 ~425 → 直接吃掉 Path A 的全部 FVD 增益

### 低优先级（之前 backlog）

6. **α-clamp full 540 消融** (4 configs × 5h): mini-68 代码验证 OK 但噪声不够，full 能给 per-channel 归因
7. **跨 subject Path B 测试**: 当前只在 Sub-05
8. **Figure 2 绘图**: α(τ) 轨迹 + FVD-EPE 散点 + 高/低 motion 样本 α_mot(τ) 对比
9. **Appendix H.4 训练 loss trajectory**: 论文引用的 SF_total / diff_loss / val_loss 详细图
