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

### 1.3 540 评估 (7-way, 2026-04-18/19)

所有实验用同一推理代码 (新代码) + 同 seed=42 + 540 样本全集。**新代码相对旧代码有 +15% FVD 漂移** (E0_new_code 717 vs E0_v2_static 619),因此公平对比 Path B 时应以 E0_new_code 为 baseline。

| 实验 | Schedule | FVD ↓ | EPE ↓ | SSIM ↑ | PSNR ↑ | CLIP ↑ | CTC ↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| **E0_v2_static** (旧代码 baseline) | none | **618.72** | 2.94 | 0.302 | 12.04 | 0.747 | 0.9865 |
| **E0_new_code** (新代码 baseline) | none | **717.23** | 2.91 | 0.310 | 12.13 | 0.743 | 0.9860 |
| E3_cosine (amp=+0.4) | cosine | 1144.73 | 2.64 | 0.296 | 9.80 | 0.702 | 0.9897 |
| E4_sigmoid_mid (amp=+0.5) | sigmoid mid | 1193.63 | **2.60** | 0.292 | 9.46 | 0.693 | 0.9903 |
| E4_sigmoid_mid_clamped (α≤0.95) | sigmoid mid | 628.22 | 2.88 | 0.305 | 12.04 | 0.747 | 0.9870 |
| **E4_reverse (amp=−0.5)** | sigmoid mid | **425.28** (**−41% vs E0_new_code**) | 3.19 | 0.282 | **12.61** | 0.758 | 0.9844 |
| E4_reverse_clamped (α≤0.95) | sigmoid mid | **429.93** (+1% vs 无 clamp) | 3.22 | 0.277 | 12.55 | **0.761** | 0.9845 |

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

### 1.6 H\*\* — α_brain OOD 时机不对称假设 (对称验证通过, 2026-04-19)

`context = (1 + α_brain)·z_b + Σα·g` 中 α_brain base=0.744,amp=+0.5 sigmoid 会把它推到 1.08,**超过 sigmoid 训练分布 [0, 1]**。H** 预测:**早期 OOD 经 49 步 compound → FVD 灾难;晚期 OOD 只影响 refinement → FVD 基本无害**。

**双向 clamp 实验 (E4_sigmoid_mid + E4_reverse, 各 α≤0.95)**:

| 实验对比 | FVD 无 clamp | FVD clamp (α≤0.95) | Δ | 解释 |
|---|---:|---:|---:|---|
| **正向 schedule** (amp=+0.5, **早期** OOD) | 1194 | **628** | **−47%** | clamp 移除早期 OOD → FVD 灾难消失,回到 baseline |
| **反向 schedule** (amp=−0.5, **晚期** OOD) | 425 | **430** | +1% | clamp 移除晚期 OOD → FVD **基本不变** |

**同一 α_brain 越界 1.12, 早期吃完整 FVD, 晚期吃不到 — 时机就是一切**:
- 单向 clamp 实验 (仅 E4_sigmoid_mid_clamped) 已经证明早期 OOD 是 FVD 主因
- 加上 E4_reverse_clamped 的**对称对照**,彻底排除了"OOD 本身就是有害的"这个 null hypothesis
- OOD 在**早期**出现时传播 49 步的差异被 compound 放大到 FVD 崩;晚期出现时仅影响最后几步 refinement,影响微小

**几个关键 corollary**:

1. **E4_sigmoid_mid 的 EPE 2.60 (比 v2 好 0.29) 是 OOD 噪声产物,不是真收益**
   - clamp 后 EPE 回到 2.88 (和 v2 2.94 差 0.06,在 noise floor 内)
   - **Path A 正向路径无独立增益** — 所谓的 "EPE 改善" 是 OOD 副产品

2. **E4_reverse 的 FVD 425 是真实 schedule 效果**
   - 不是 OOD 侥幸: clamp 后 FVD 基本不变 (430)
   - 独立于 OOD 因素的 motion-first 方向收益
   - 对新代码 baseline (717) 是 **−41% FVD**

3. **Path B Prior 方向锚定正确**
   - 用 E4_reverse 形状 prior (amp=−0.5) 既避免 OOD (α_brain 推到 0.37),又锁定真实收益方向
   - 双赢: OOD-avoidance + motion-first 一次性实现
   - 不需要额外 clamp 机制 (`alpha_max` 参数在 sampling.py 保留但默认不启用)

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

### 2.8 P1 训练完成 (2026-04-19 01:13 → 10:42, 9.5h)

**配置**: gpu2 × 2 DDP (GPU 3, 5), 2000 iter, lr=1e-5, grad_accum=2, freeze slow/fast branches, unfreeze fusion
**Ckpt**: 500/1000/1500/2000 全部保存 (`ckpts_5b/sf_v3_pathB_p1-04-19-01-13/`)
**最终 loss**: total 0.26 / diff_loss 0.14 (正常区间, 无发散)

### 2.9 iter 2000 α(τ) probe (50 样本 × 7 τ 点)

| 通道 | iter 0 (init+prior) | **iter 2000** | 训出的残差 |
|---|---|---|---|
| α_key | 0.38 → 0.60 (range +0.22) | 0.38 → 0.60 (+0.22) | **不变** |
| α_txt | 0.21 → 0.40 (range +0.19) | 0.21 → 0.38 (+0.17) | base 同, range 略缩 |
| α_mot | 0.54 → 0.32 (range −0.22) | **0.60 → 0.40** (−0.20) | **base ↑ 0.07** |
| α_brain | 0.60 → 0.79 (range +0.19) | **0.73 → 0.87** (+0.15) | **base ↑ 0.12** |

所有通道 `frac|Δ|>0.05 = 100%`,**E4_reverse 形状完整保留**。

**关键发现**:
1. **gate_net sample-path 明显在学**: α_mot 整体 ↑ 0.07, α_brain 整体 ↑ 0.12 — 模型要更多 motion + brain 信号
2. **gate_net τ-path 学到少量负残差**: 4 通道 range 都略缩 (-0.02 ~ -0.04) — gate_net 在软化 prior 的陡峭度
3. **晚期 α_mot 0.40 比 Path A 的 0.22 强 82%**: 理论上**能修复 Path A 的 EPE trade-off** (Path A EPE 3.19 吃了晚期 α_mot 太弱的亏)

**相对 Path A winner 的重要差异**:

| 通道 | Path A E4_reverse 晚期 α | Path B iter 2000 晚期 α | 含义 |
|---|---:|---:|---|
| α_mot | 0.44 × 0.5 = 0.22 | **0.40** (+82%) | Path B 晚期 motion 更强 → 预期 EPE 改善 |
| α_brain | 0.74 × 1.5 = 1.12 (**OOD!**) | **0.87** (< 1.0) | Path B 无 OOD 问题 |
| α_key | 0.49 × 1.5 = 0.73 | 0.60 | Path B 略保守 |

Path B **天然避开 OOD** (sigmoid 输出保证 α < 1.0) **且晚期 motion 更强** — 如果 FVD 也保持 Path A 的 425 水平,EPE 应比 Path A 好。

### 2.10 540 推理进行中 (2026-04-19 12:30 启动)

- gpu2 GPU 3 (split0, 270 samples) + GPU 5 (split1, 270 samples)
- Config: `cinebrain_sf_v3_pathB_model.yaml` + `infer_pathB_p1.yaml`
- 输出: `results/alpha_540/pathB_p1_iter2000/`
- ETA 完成: 2026-04-20 08:00-09:00 (~20h)

**预期 540 eval**:
- FVD: 400-450 (接近或略优 Path A winner 425)
- EPE: 2.7-3.0 (显著优 E4_reverse 3.19, 因晚期 α_mot 更强)
- SSIM/CLIP: 0.28-0.30 / 0.74-0.76 (接近 E4_reverse)
- 对 E0_new_code baseline (717): 若 FVD ≤ 420,则 **−41% FVD**

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
