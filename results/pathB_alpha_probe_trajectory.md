# Path B α(τ) Probe 学习轨迹

## 实验目的
诊断 Path B 训练期间 gate_net 学到的 α(sample, τ) 是如何从 init（prior 形状）偏移的，跨 5 个训练检查点（iter 0/500/1000/1500/2000）。配合 14-metric 评估数据提供"为什么 iter 2000 被 Pareto-dominate"的微观证据。

## 实验设置
- **ckpts**: `ckpts_5b/sf_v3_pathB_p1-04-19-01-13/{500,1000,1500,2000}/` + iter 0 (partial_load 起点，带 E4_reverse prior 的 pathB_init)
- **Probe 脚本**: `tools/probe_pathB_alpha_curve.py`
- **样本数**: 50 held-out (`/public/home/maoyaoxin/zhangt/xxt/datasets/sub-0005_test_va.json` 前 50)
- **τ 网格**: [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] (7 点)
- **日期**: 2026-04-22
- **Output**:
  - `results/pathB/alpha_curve_init_with_prior_v2.json` (iter 0)
  - `results/pathB/alpha_curve_p1_iter{500,1000,1500}.json`
  - `results/pathB/alpha_curve_p1_2000_iter2000.json`

## 关键表: α(τ) 均值轨迹 (50-sample average)

### 每通道 (τ=0) / (τ=1) / 均值漂移

| 通道 | iter 0 | 500 | 1000 | 1500 | 2000 | 漂移模式 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **α_key** avg(0,1) | 0.490 | 0.479 | 0.486 | 0.490 | 0.489 | **基本不动** — gate_net 认可 prior |
| **α_txt** avg | 0.306 | 0.235 | 0.273 | 0.303 | 0.297 | **非单调探索** — 500 时探至低位，1500 回归 prior |
| **α_mot** avg | 0.428 | 0.474 | 0.473 | 0.495 | **0.503** | **单调 +0.075** — motion 信号持续增强 |
| **α_brain** avg | 0.693 | 0.758 | 0.775 | 0.789 | **0.799** | **单调 +0.106** — brain 信号最大漂移 |

### Range（α(τ=0) - α(τ=1)）随训练变化

| 通道 | iter 0 | 500 | 1000 | 1500 | 2000 | Range 变化 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| α_key | +0.224 | +0.223 | +0.222 | +0.221 | +0.221 | -1.3% |
| α_txt | +0.186 | +0.152 | +0.167 | +0.177 | +0.174 | -6.5% |
| α_mot | -0.218 | -0.217 | -0.211 | -0.204 | -0.201 | -8.1% |
| α_brain | +0.189 | +0.161 | +0.155 | +0.149 | +0.145 | **-23.2%** |

### L2 距离（距 iter 0 初始化）

| 通道 | iter 500 | iter 1000 | iter 1500 | iter 2000 |
|:---:|:---:|:---:|:---:|:---:|
| α_key | 0.030 | 0.009 | 0.003 | 0.003 |
| α_txt | 0.189 | 0.087 | 0.011 | 0.026 |
| α_mot | 0.124 | 0.121 | 0.182 | 0.203 |
| α_brain | 0.177 | 0.223 | 0.260 | **0.286** |

## 与 TB 训练 loss 对照

从训练 log (`logs/pathB_p1_2000.log`, eval_interval=500) 的 validation loss 轨迹：

| Iter | 主 val loss | sf_total | L_fast | L_distill_spatial | diff_loss |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 500 | 0.434 | 0.296 | 96.8 | 410 | 0.131 |
| 1000 | 0.417 | **0.235** ✓ | **76.4** ✓ | **323** ✓ | 0.142 |
| **1500** | **0.396** ✓ | 0.265 | 86.2 | 366 | 0.148 |
| 2000 | **0.486** ↑ | **0.325** ↑ | **106** ↑ | **461** ↑ | **0.117** ↓ |

**读出**:
- 主 val loss iter 1500 最低
- SF 辅助损失 (L_fast / L_distill_spatial) iter 1000 最低
- diff_loss (DiT 重建) 单调改善至 iter 2000
- iter 2000 在主 val loss 和 SF loss 上双双爆恶化 → **gate_net 过训练 + DiT 与 SF 目标互撕**

## 核心发现

### Finding 1: gate_net 保留 prior 形状，仅平移 base level
所有 5 个 iter 上所有通道的 `frac|Δ|>0.05 = 100%`（50 样本全部维持 prior 所需最小 range）。iter 2000 时 range 仅缩 -1% to -23%，说明 prior 的骨架被完整保留。

### Finding 2: gate_net 主要学的是 "(sample) base shift" 而非 "(τ) 再形状"
α_mot 和 α_brain 的 base 漂移占 L2 总变化的绝大部分；range 收缩贡献小。即 gate_net 的 sample-path（pooled feature）在学 per-sample offset，t_emb path 几乎只是维持 prior。

### Finding 3: α_brain 漂移与 FVD 退化高度相关
α_brain +0.106 over 2000 iter 正好对应 FVD 从最低点 (iter 1000: 501) 反弹到 iter 2000 的 518。E4_reverse_clamped 实验（clamp α_brain 到训练分布）表明 α_brain 的过大是 FVD 敏感的主要漂移方向。这为 Path B Phase 2 正则设计（对 α_brain 施加 L2 penalty）提供先验证据。

### Finding 4: α_txt 非单调证明 gate_net 有回归能力
α_txt 在 iter 500 下探至 0.235，iter 1500 回到 0.303。说明 gate_net 能把"一度偏离的通道"拉回 prior，α_mot/α_brain 的持续漂移不是优化器病态，是被 loss 一致拉向的真实最优。

## 关联数据源
- α(τ) JSON: `results/pathB/alpha_curve_*.json`
- 训练 log: `logs/pathB_p1_2000.log`
- 14 指标评估: `results/pathB_full14_results.md`
- 原 probe 脚本: `tools/probe_pathB_alpha_curve.py`
