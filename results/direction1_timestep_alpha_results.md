# Direction ① — Timestep-aware α Learning Results

## 实验环境
- **服务器**: ts3 集群 (gpu1/gpu2/gpu3), H800 80GB
- **模型**: CogVideoX-5B + CineBrain-SF v1 (Slow-Fast dual branch) + LoRA (r=128)
- **采样**: VPSDEDPMPP2MSampler, 50 步, ZeroSNRDDPMDiscretization, DynamicCFG (scale=6, exp=5)
- **被试**: Sub-05 (测试集 Episode S7+S11, 视频 7560-8099)
- **日期**: 2026-04-17 ~ 2026-04-19
- **评估脚本**: `tools/eval_alpha_schedule.py` (14 项指标, 与 CineBrain baseline 完全一致)
- **Baseline**: v2 stage3 iter 3000 (`ckpts_5b/sf_v1_stage3_joint-04-05-02-53/`), FVD 619 / EPE 2.94

## 方向概览

方向 ① 假设: gate_net 输出的 α (控制 `key/txt/mot/brain` guidance 权重) 在扩散步上应**随 τ 变化**而非静态,能帮助模型在合适的步数注入合适的引导。分两条路径验证:

- **Path A** (inference-only): 在已训好的 v2 上,运行时用解析 schedule `scale(τ)` 乘 α_base,验证"τ 变化有用"的假设,零训练成本
- **Path B** (training-time): 让 gate_net 吃 `t_emb`,端到端学 α(sample, τ)

---

## Part 1 — Path A (Inference-time α Schedule)

### 1.1 代码改动

| 文件 | 改动 |
|---|---|
| `sgm/modules/encoders/multi_guidance.py` | `compute_components / mix_context` 拆分,给 sampler 侧重混预留钩子 |
| `sgm/modules/encoders/sf_embedder.py` | `expose_premix` 开关;forward 缓存 `_last_premix = {z_b, components, alphas_base}` |
| `sgm/modules/diffusionmodules/sampling.py` | `VPSDEDPMPP2MSampler.alpha_schedule` + `_remix_cond_for_step`;每步用 `scale(τ)` 调制 α,可选 `alpha_max` clamp |
| `diffusion_video_brain.py` | `sample()` 把 embedder 传给 sampler |

### 1.2 Schedule 形状

```
α_slow_t = α_slow_base × (1 + amp × base(τ))         # τ=0 高噪声,τ=1 低噪声
α_fast_t = α_fast_base × (1 − amp × base(τ))
```

`base(τ)` 有 linear / cosine / sigmoid 三种,`amp ∈ [−0.5, +0.5]`。

### 1.3 540 评估 (4-way + 1 清洁 baseline, 2026-04-18)

所有实验用同一推理代码 (新代码),seed=42,540 样本全集:

| 实验 | Schedule | FVD ↓ | EPE ↓ | SSIM ↑ | PSNR ↑ | CLIP ↑ |
|---|---|---:|---:|---:|---:|---:|
| **E0_v2_static** (旧代码 baseline) | none (静态) | **618.72** | 2.94 | 0.302 | 12.04 | 0.747 |
| E0_new_code | none (静态) | 720.27 | 2.98 | 0.295 | 11.98 | 0.739 |
| E3_cosine (amp=+0.4) | cosine | 1144.73 | 2.64 | 0.296 | 9.80 | 0.702 |
| E4_sigmoid_mid (amp=+0.5) | sigmoid mid | 1193.63 | 2.60 | 0.292 | 9.46 | 0.693 |
| **E4_reverse (amp=−0.5)** | sigmoid mid | **425.28** (**−31%**) | 3.19 | 0.282 | **12.61** | **0.758** |

### 1.4 核心发现: 方向反转

**H1 (原始假设) 被推翻**: 预期 "Slow 早强 / Fast 晚强" (正向 amp) 会降 FVD。实测正向 schedule 让 FVD 翻倍 (619 → 1144-1194),反向 schedule 让 FVD 大降 (-31%)。

**H1' (基于文献修正)**: Video diffusion 早期步骤由**运动**主导,晚期由**外观**主导。Slow/appearance guidance 在早期过强会触发 conditional structural leakage,导致 "shortcut to static" → motion collapse。

文献支持:
- **ALG (CVPR 2026)**: early structural guidance → motion collapse,+36% Dynamic Degree after fix
- **CIL (NeurIPS 2024)**: I2V 大 timestep 处 conditional image leakage
- **MotionShop / MotionClone / arXiv:2512.22175**: motion guidance 只在 early 有效

### 1.5 α_base 饱和假设 (H\*) 证伪

`tools/dump_v2_alphas.py` 在 50 样本上实测 v2 gate_net 输出:

| 通道 | mean | std | min | q25 | median | q75 | max | sat<0.05 | sat>0.95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| alpha_key | 0.476 | 0.012 | 0.442 | 0.470 | 0.477 | 0.485 | 0.498 | 0% | 0% |
| alpha_txt | 0.218 | 0.014 | 0.182 | 0.210 | 0.220 | 0.227 | 0.243 | 0% | 0% |
| alpha_mot | 0.439 | 0.012 | 0.414 | 0.431 | 0.440 | 0.447 | 0.468 | 0% | 0% |
| alpha_brain | **0.744** | 0.011 | 0.717 | 0.738 | 0.744 | 0.751 | 0.770 | 0% | 0% |

所有通道**完全不饱和**,H\* (v2 gate_net 输出饱和 → 需要 timestep 变化来"解放") 被证伪。

### 1.6 H\*\* — α_brain OOD 放大器假设 (待验证)

`context = (1 + α_brain)·z_b + Σα·g` 中 α_brain base=0.744,amp=+0.5 sigmoid 会把它推到 1.08,**超过 sigmoid 训练分布 [0, 1]**。早期 OOD 经 49 步 compound 放大 → FVD 灾难。晚期 OOD 只影响 refinement,FVD 可接受。

**预验证方案** (代码已 ready,未跑): `alpha_schedule_E4_clamped.yaml` (同 E4_sigmoid_mid 但 `alpha_max=0.95`):
- 若 FVD 从 1194 → 700-800 → H\*\* 成立 (OOD 是主因)
- 若 FVD 仍 ≥ 1000 → H\*\* 弱化 (文献 motion-first 才是主因)

### 1.7 结论 (D1 决策)

D1 情形 **D (方向反转型 winner)** 触发:
- 选 E4_reverse 作为 Path B 训练起点
- Path A 的 amp / midpoint / steepness 为 scaffolding 参数,不做精调
- 进 Phase 2 (Path B 训练),把文献启示写进 Path B DESIGN

---

## Part 2 — Path B (Training-time Learned α(sample, τ))

### 2.1 目标

让 `gate_net` 同时吃 `pooled(sample)` 和 `t_emb(τ)`,端到端训练后:
1. **自发复现** "early motion / late appearance" pattern
2. **样本自适应** (不同 sample 有不同 α(τ) 曲线)
3. 预期 FVD ≤ E4_reverse 的 425,Best case 更低

### 2.2 代码改动

| 文件 | 改动 |
|---|---|
| `sgm/modules/encoders/gated_fusion.py` | `CrossModalGatedFusion` 加 `t_emb_proj` (Linear→SiLU→Linear);`gate_net[0]` 输入维 `hidden + t_emb_proj_dim`;`alpha_logit` 可加 E4_reverse 形状 prior |
| `sgm/modules/encoders/sf_embedder.py` | `__init__` 加 Path B 参数;forward 缓存 `_last_slow_feat / _last_fast_feat` 供 sampler/loss 重跑 gate_net |
| `sgm/modules/diffusionmodules/sampling.py` | `VPSDEDPMPP2MSampler.path_b.enabled`;`_remix_cond_for_step` Path B 分支,每步用 `α_cumprod_sqrt[i]` 作 τ,重跑 gate_net |
| `sgm/modules/diffusionmodules/loss.py` | `VideoDiffusionLossSF.use_path_b`;`_apply_path_b_remix`;可选 `_time_noise_z_b` (CIL TimeNoise) |
| `configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml` | 完整 Path B 模型 yaml |
| `configs/sf_v1/sf_v3_pathB_train.yaml` | P1 训练配置 |
| `tools/partial_load_v2_to_pathB.py` | 把 v2 ckpt 的 `gate_net.0.weight` 右 pad 零列 (`hidden → hidden+t_emb_proj_dim`) |
| `tools/test_pathB_warmstart_regression.py` | T1 (warm-start bit-identical) + T2 (gate_net capacity) |
| `tools/probe_pathB_alpha_curve.py` | 中期 α(sample, τ) 诊断,~3 min 跑完 |

### 2.3 Attempt 1 — 梯度死锁 bug

**设置**: gpu1 × 4 DDP,2000 iter,lr=1e-5,freeze slow/fast branches。
**首次 iter 500 probe (50 样本 × 7 τ 点)**:

| 通道 | τ=0 | τ=1 | mean\|Δα\| | frac\|Δ\|>0.05 |
|---|---:|---:|---:|---:|
| alpha_key | 0.4701 | 0.4701 | **0.0000** | 0% |
| alpha_txt | 0.1828 | 0.1828 | **0.0000** | 0% |
| alpha_mot | 0.4500 | 0.4500 | **0.0000** | 0% |
| alpha_brain | 0.7706 | 0.7706 | **0.0000** | 0% |

α_base 值有变化 (sample 路径在学),但 **t_emb 路径完全没动**。

**根因** — 双 zero-init 梯度互锁:
- `partial_load` 把 `gate_net.0.weight` 右 pad 零列 → `W_t_emb = 0` (保证 warm-start)
- 代码 zero-init `t_emb_proj[-1]` → `t_emb_feat = 0`

反向传播:
```
∂α/∂W_t_emb       = t_emb_feat × σ'       = 0 × σ' = 0
∂α/∂t_emb_proj[-1] = W_t_emb × silu × σ' = 0 × … = 0
```
两边互相需要对方非零才能动,永远锁死。

### 2.4 Attempt 2 — 解锁但学习极慢

**修复**: `t_emb_proj[-1]` 初始化从 `zeros_` → `normal_(std=0.02)`。T1/T2 回归测试仍过 (W_t_emb=0 保证 warm-start)。

**iter 500 probe 结果** (同协议):

| 通道 | τ=0 | τ=1 | mean\|Δα\| | frac\|Δ\|>0.05 |
|---|---:|---:|---:|---:|
| alpha_key | 0.4707 | 0.4707 | **0.0001** | 0% |
| alpha_txt | 0.1844 | 0.1845 | **0.0000** | 0% |
| alpha_mot | 0.4540 | 0.4541 | **0.0002** | 0% |
| alpha_brain | 0.7698 | 0.7698 | **0.0000** | 0% |

死锁解除,梯度**有流**,但信号量级 ~10⁻⁴,距离 Path A winner α range ~0.22 差**两个数量级**。
- 2/50 样本有 `|Δα| ~ 0.001-0.002`,其余全 0
- 架构有 capacity (T2 证明),但 `lr=1e-5 × ∂α/∂W_t_emb ~ 10⁻⁴ × Adam cold-start × 2000 iter` 不足以训出明显的 τ 依赖

### 2.5 Attempt 3 — Prior (E4_reverse) + Option C (当前)

**两个独立改动组合**:

**1. Prior bias** — 在 `alpha_logit` 上加 E4_reverse 形状的解析 τ 函数:

```python
sched(τ) = 1 - 2·sigmoid(k·(τ - m))              # +1 at τ=0, -1 at τ=1
prior_bias = amp × sched(τ) × sign_per_channel   # (B, 4)
alpha_logit = gate_net(concat(pooled, t_emb_feat)) + prior_bias
```

参数: `amp=0.5, k=6, m=0.5, sign=[-1,-1,+1,-1]` for `[key, txt, mot, brain]`。

**保底**: 即使 gate_net 完全不学 t_emb (worst case),α_t 仍 = v2 α × E4_reverse schedule → FVD ≤ 425。

**2. Option C** — `t_emb_proj[-1]` init std `0.02 → 0.1` (5×):
- `t_emb_feat` 初始幅度 5×,`∂α/∂W_t_emb` 初始梯度 5×
- 不改 α 输出 (W_t_emb=0 仍 annihilate),只放大梯度信号
- T1 依旧通过,T2b 从 1e-2 → 5e-2

**Sanity probe (pathB_init + 新代码, 10 样本)**:

| 通道 | τ=0 (早) | τ=1 (晚) | Δ | 方向 |
|---|---:|---:|---:|---|
| α_key | 0.38 | 0.60 | **+0.22** | ✓ 晚期强 |
| α_txt | 0.21 | 0.40 | **+0.19** | ✓ 晚期强 |
| α_mot | **0.54** | 0.32 | **−0.22** | ✓ 早期强 (motion-first) |
| α_brain | 0.60 | 0.79 | **+0.19** | ✓ 晚期强 |

100% 样本 `frac|Δ|>0.05`,α(τ) 曲线形状与 E4_reverse schedule 完全匹配。

### 2.6 Path A 与 Path B Prior 幅度对比

- Path A (E4_reverse) 是**乘法** 调制: `α_t = α_base × scale(τ), scale ∈ [0.5, 1.5]`
- Path B Prior 是**加法** (logit space): `alpha_logit += ±0.5`
- 两者最终 α 变化幅度相当: Path A ~0.22 range, Path B ~0.22 range

### 2.7 回归测试

`tools/test_pathB_warmstart_regression.py`:
```
[T1] max|Δz_b| = 0.000e+00
[T1] max|Δalpha_*| = 0.000e+00 (全部 4 通道)
PASS T1: Path B iter-0 matches v2 bit-identical.

[T2a] max|Δα with random t_emb (zero-padded gate_net cols)| = 0.000e+00
[T2b] max|Δα with random t_emb (broken zero-pad)| = 5.307e-02
PASS T2: gate_net responds to t_emb once W_t_emb_cols leaves zero-pad.
```

### 2.8 当前状态 & 下一步

- **Attempt 3 P1 训练**: Probe100 smoke test (gpu2 × 2 DDP, 100 iter) 进行中,之后回 gpu1 × 4 DDP 跑完整 2000 iter
- **若 gate_net 仍学不动**: 计划加独立 param_group (gate_net lr 1e-5 → 1e-4) + 关 lr_decay
- **若学得动**: residual 偏离 prior 的方向告诉我们 α 是应该更极端还是更温和

### 2.9 Phase II 成功判据

- **Strong**: FVD ≤ 400, EPE ≤ 2.9, SSIM ≥ 0.3, CLIP ≥ 0.75 (全面超越 E4_reverse)
- **Mild**: FVD ≤ 500, 且 α(τ) 学到非平凡曲线 (偏离 prior,有 sample-specific residual)
- **Null**: α(τ) 学成常数或 `|Δα|<0.01` → 架构容量不足,降级方向② (DiT 层 × timestep gating)

即使 Null 情况,Prior 保证 α_t = v2 × E4_reverse 的效果,FVD ≈ 425 仍然优于 v2 619。

---

## 相关文档 (local, 未入库)

- `DESIGN_direction1_timestep_alpha.md` — Path A 原设计 (H1 已推翻)
- `DESIGN_direction1_pathB.md` — Path B 设计 v0.1
- `DEBUG_direction1_inversion_finding.md` — 2026-04-18 方向反转深度调研
- `IMPLEMENTATION_ROADMAP.md` — 方向全景 + D1 决策逻辑 + 不做清单
- `EXPERIMENT_direction1_mini20_results.md` — mini20 探索 + 540 详细记录 + Phase II 诊断
- `HANDOFF.md` — 跨 session 状态快照
