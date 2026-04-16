# Code Review Response — CineBrain-SF v1

**日期**: 2026-04-05（v2，根据 reviewer 二次反馈更新）
**对应 Review**: `REVIEW_REPORT.md` (2026-04-04)
**处理方式**: 逐项验证 → 独立判断 → 分级处理

---

## 处理原则

1. 每个 issue 都对照实际代码验证，不盲目接受
2. 区分"代码 bug"和"方法设计选择"——前者修，后者评估后决策
3. 区分"当前影响训练"和"潜在风险"——前者优先
4. 不因为 review 的存在就推翻已有的实验进展

---

## Top Issues 逐项回应

### Issue #1: LoRA alpha 未设置，scaling = 1/128

**验证结果**: 确认存在。`lora2.py:83` — `self.scaling = self.lora_alpha / self.r`，默认 `lora_alpha=1`，`r=128`，scaling=0.0078。

**我们的判断**: reviewer 对 bug 本身的判断完全正确。但"Stage 3 所有 loss 完全平坦"的结论与实际数据不符：
- Valid L_guide: 0.200 → 0.179（↓10%）
- Valid L_struct: 0.027 → 0.012（↓56%）
- Valid L_temp_delta: 0.060 → 0.037（↓38%）

这些改善来自 Fast Branch 解冻 + Fusion 继续训练，而非 DiT LoRA。LoRA scaling=1/128 确实让 DiT 几乎不更新，但其他可训练模块在正常工作。

**修复**: `lora_alpha: 128`，使 scaling=1.0。**已应用。**

**叠加效应注意**: Fix 1 (scaling 1/128→1.0) + Fix 4 (grad_clip 0.1→1.0) 的叠加使 LoRA 有效学习信号放大约 1280 倍。这是从"几乎不更新"到"正常更新"的跳跃。新训练前 100 iter 需密切观察，如果出现 loss spike 或 NaN，优先降 lr（1e-4 → 5e-5），不回退修复。

---

### Issue #2: 动态缩放使所有 SF Loss lambda 权重完全失效

**验证结果**: 确认存在。`loss.py:320-323` — `scale = diff_loss.detach() / (sf_total.detach() + 1e-8); sf_total = sf_total * scale`。

**我们的判断**: reviewer 的核心观点正确，但"所有 lambda 失效"过于绝对。实际效果：
- sf_total **内部**各 loss 的相对权重（lambda 比例）仍然有效
- 但 sf_total 的**绝对量级**被强制拉到与 diff_loss 相同
- 更严重的问题是：**冻结模块的 loss 稀释了可训练模块的梯度**。Stage 2 中 sf_total ≈ L_fast(30) + L_slow(0.2) + L_guide(0.16) ≈ 30.36，scale = 0.26/30.36 = 0.0086，L_guide 的有效梯度被缩小约 190 倍

但 Stage 2 训练并未完全失效——GatedFusion 学会了有意义的 alpha 分化（brain 0.94, txt 0.83），这主要靠 L_diff 梯度穿透 DiT 传回来的信号。

**修复**: 删除动态缩放，改用固定 `lambda_sf=0.003`。

**lambda_sf 取值依据**: 真实 unscaled sf_total ≈ 80-113（由 validation 日志确认——train 日志中 `sf/total` 显示的 ~30 是动态缩放后的假象，因为缩放只在 `requires_grad=True` 即训练时触发，validation 在 `no_grad` 下不触发）。0.003 × 100 = 0.3，与 diff_loss ≈ 0.26 同量级。**已应用。**

**补充说明**: 动态缩放的 train/val 行为不一致也是 train-val gap 300x 的**真正原因**（非 EMA，非方差）——train 时缩放触发（total ≈ 0.26 + 0.13 = 0.39），val 时不触发（total ≈ 0.26 + 80 = 80）。删除后 gap 将大幅缩小。

**新训练诊断**: 已在 `loss.py` 中添加 debug logging，每个 iter 记录 `debug/raw_sf_total`、`debug/diff_loss`、`debug/sf_diff_ratio`。如果 sf_diff_ratio 稳定在 1-3 则 lambda_sf 合理，如果 >5 需降到 0.001-0.002。

---

### Issue #3: fMRI 和 EEG 原始数据完全没有归一化

**验证结果**: 代码层面确认 `data_video.py` 无归一化逻辑。

**数据验证（已完成）**: 直接检查了 .npy 文件统计量：

```
fMRI (4 samples):
  mean=0.75~0.99, std=0.80~0.83, range=[-2.2, 3.7], shape=(18946,)
  → 已经过归一化（近似 z-score，但 mean 偏移 ~0.8-1.0，非 raw BOLD）

EEG (4 samples):
  mean≈0.0000, std=0.004~0.010, range≈[-0.004, 1.0], shape=(69, 800)
  → 已经过归一化（min-max 或截断归一化，max=1.0）
```

**结论**: Issue #3 **不成立**。数据已在上游预处理中归一化。fMRI mean 偏移 ~0.9 值得注意但不构成训练障碍（encoder 的 learned linear projection 可以补偿 bias）。

**处理**: **不修改**。

---

### Issue #4: Guidance 注入方式 — 全局向量广播 + alpha_brain 主导

**验证结果**: 代码确认。`multi_guidance.py:73-95` — guidance 向量 `(B, D)` 通过 `.unsqueeze(1)` 广播到所有 226 个 token。alpha_brain=0.94 在 Stage 2 评估中确认。

**我们的判断**: 这**不是 bug，而是 GatedFusion 学到的结果**。模型发现 brain latent 对去噪最有用就提高了权重。修复 LoRA alpha 后 DiT 能真正适配 brain conditioning，alpha 分布可能自然调整。

reviewer 建议改用 cross-attention 注入是**方法层面改进**，不是 bug fix。

**处理**: **延后到后续版本**。当前优先修复 P0 级 bug，方法改进排在消融实验之后。

---

### Issue #5: 226 brain tokens 远超 DiT 预训练的 text 长度

**验证结果**: 正确。CogVideoX 预训练 text tokens ≤77，现在 226 个。

**我们的判断**: 这是 CineBrain/CineSync 的**原始架构决策**，不是 SF v1 引入的。CineSync 在此设计下能工作。LoRA alpha 修复后 DiT 的适配能力将大幅增强，无需现在改。

**处理**: **不修改**。记录为后续优化方向（learned pooling 降到 64 tokens）。

---

## P0 级问题回应

### P0-4: gt_dyn_label_2class 键名不匹配

**验证结果**: 确认。`data_video.py:674` 映射为 `gt_dynamics_class`，`sf_losses.py:188` 查找 `gt_dyn_label_2class`。L_dyn 永远不计算。

**修复**: 在 `data_video.py` 的 target 映射后添加 3-class→2-class 转换。**已应用。**

### P0-5: Text caption 随机截断与固定 gt_text_embed 不匹配

**验证结果**: 确认。`data_video.py:628-629` 每次随机截取 2 句，gt_text_embed 是固定提取的。

**我们的判断**: 影响有限。L_txt 在训练中稳定下降（0.20→0.19），说明噪声未严重影响学习。

**处理**: **暂不修改**，记录为后续优化项。正式实验前统一 caption 处理。

### P0-6: _clip_loss 缺少 L2 归一化

**验证结果**: 确认。`sf_embedder.py:270-273` 无 `F.normalize`。

**我们的判断**: reviewer 也正确指出当前 `mode: infer` 时不触发。**目前不影响训练。**

**处理**: **暂不修改**。如果后续启用 clip loss mode，必须先修复。

---

## P1 级问题回应

### P1-1: Stage 3 所有 loss 960 iter 完全平坦

**我们的判断**: **结论不准确**。reviewer 可能分析的是早期数据。我们的完整 Stage 3 validation（5 个 checkpoint）显示多项指标持续改善（L_guide ↓10%, L_struct ↓56%, L_temp_delta ↓38%），只是 DiT LoRA 因 scaling=1/128 几乎无贡献。修复 LoRA alpha 后应有更大改善。

### P1-3: gradient_clipping=0.1 过于激进

**我们的判断**: 合理怀疑。与 LoRA scaling 叠加后有效信号被截断的概率很高。

**修复**: Stage 3 config 提高到 1.0。**已应用。**

### P1-4: 无 LR warmup / scheduler

**验证结果**: **不准确**。SAT 框架有内置的 lr warmup 逻辑，从训练日志可以看到 lr 从 5e-6 逐步升到 ~9.8e-5 再 cosine decay。reviewer 可能只看了 yaml 配置没看实际 lr 日志。

**处理**: **不修改**，warmup 已存在。

### P1-5: AlignmentLoss 在 bs=1 下恒为 0

**我们的判断**: 已知问题，HANDOFF.md 和设计文档中已记录。P0 蒸馏方案就是为了绕开此问题。

**处理**: **不修改**，lambda 已设为合理值。长期方案：queue-based negatives（已列入后续路线）。

### P1-6: L_gm (motion guidance loss) 完全缺失

**验证结果**: 确认。`GuidanceLoss.forward()` 只计算 L_gk 和 L_gt，L_gm 从未实现。

**处理**: **暂不修改**。motion guidance 的直接监督需要定义合理的 target（当前没有合适的 motion ground truth 可用于 guidance consistency 计算）。记录为后续项。

### P1-7: Train-Valid gap 达 300-400 倍

**根因**: 动态缩放的 train/val 行为不一致。train 时 `requires_grad=True` 触发缩放（sf_total 从 ~80 压缩到 ~0.13），val 时 `no_grad` 下 `requires_grad=False` 不触发缩放（sf_total 保持 ~80）。这导致 train total ≈ 0.39 vs val total ≈ 80，即 ~200x gap。不是方差问题，不是 EMA，是计算公式在 train/val 下不同。

**处理**: Fix #2（删除动态缩放）已解决根因。新训练中 train 和 val 使用相同的 `lambda_sf` 缩放，gap 将大幅缩小。后续 eval_iters 提高到 10+。

### P1-8: P1 训练中 P0 蒸馏知识被严重遗忘

**我们的判断**: 这是**设计预期**，不是问题。P1 的目标就是让 EEG 特征脱离 fMRI 空间（cosine ≈ 0），蒸馏 loss 上升是 Fast/Slow 正交化的必然代价。设计文档 §5.5 明确描述了 λ_distill staged decay。

### P1-9: GatedFusion 先 concat 再投影 + 只取 slow-aligned tokens

**验证结果**: 确认。`gated_fusion.py:71,85` — concat 后投影，只取前 S 个 token。

**我们的判断**: 已知技术债（HANDOFF.md "M-01"）。影响 Fast Branch 信息在 fusion 中的表达力。

**处理**: **延后到方法改进阶段**。

### P1-10: Delta loss 第一帧恒为 0

**验证结果**: 正确。`delta = z_t - z_1`，t=1 时恒为 0。

**处理**: 影响微小（1/9 帧），**暂不修改**。

### P1-11: Causal mask 使早期 temporal query 孤立

**验证结果**: 正确，t=1 的 self-attention 只能看到自己。但 cross-attention 仍能看到全部 226 个 EEG tokens，信息来源并不匮乏。

**我们的判断**: reviewer 说"考虑只在 cross-attention 中保持因果约束，self-attention 使用 full attention"——**方向反了**。cross-attention 不需要因果约束（EEG tokens 不是时序排列的），self-attention 才需要（temporal queries 之间的因果关系）。当前实现是正确的。

**处理**: **不修改**。如果 Stage 3 评估中 temporal 指标恶化，考虑关闭 causal mask 做消融。

### P1-12: Frozen 模块 loss 仍参与 sf_total 并影响动态缩放

**我们的判断**: **已通过 Fix #2 解决**。删除动态缩放后，frozen loss 的梯度不会传播到冻结参数（requires_grad=False），不影响可训练模块。lambda_sf 统一控制 sf_total 对总 loss 的贡献比例。

### P1-13: Gating alpha_mot 动态分化退化

**我们的判断**: 在 Stage 2（frozen DiT + frozen branches）下 alpha_mot 分化消失是预期的——DiT 无法区分 motion guidance 的好坏。Stage 3 修复 LoRA alpha 后，DiT 能真正学习利用 guidance，alpha_mot 分化有望恢复。

---

## P2 级问题回应

| # | 判断 | 处理 |
|---|------|------|
| P2-1 TCN 在 pool 后运行 | 正确。EEG encoder 架构限制 | 延后（需改 encoder 架构） |
| P2-2 缺 final LayerNorm | 正确 | 延后（影响小） |
| P2-3 TemporalAttentionPool bf16 | 已知，temporal_dynamics.py 用了 BF16SafeAttention 规避 | 不修改 |
| P2-4 StructureHead 354M 参数 | 正确但 StructureHead 当前未启用 | 不修改 |
| P2-6 Sigmoid 无竞争 | 设计选择。Sigmoid 允许多个 alpha 同时高 | 不修改（后续可实验 softmax） |
| P2-7 ucg_rate 在 branch_pretrain 生效 | 正确 | 记录为优化项 |
| P2-13 L_struct 含对角线 | 正确 | **已修复**（off-diagonal mask） |
| P2-17 无 gradient norm 记录 | 正确，对新训练诊断至关重要 | **已添加** debug logging（见下方） |

---

## 已应用的修复清单

| # | 修复 | 文件 | 说明 |
|---|------|------|------|
| Fix 1 | LoRA alpha: 128 | `cinebrain_sf_v1_model.yaml` | scaling 从 1/128 → 1.0 |
| Fix 2 | 删除动态缩放 | `loss.py` | 改用固定 `lambda_sf=0.003` |
| Fix 3 | gt_dyn_label_2class | `data_video.py` | 3-class→2-class + 正确键名 |
| Fix 4 | gradient_clipping | `sf_v1_stage3_joint.yaml` | 0.1 → 1.0 |
| Fix 5 | L_struct off-diagonal | `sf_losses.py` | 去掉无效的对角线计算 |
| Fix 6 | Debug logging | `loss.py` | 记录 raw_sf_total / diff_loss / sf_diff_ratio |

## 已完成的验证

| 项目 | 结论 |
|------|------|
| fMRI 数据归一化 | 已预处理（mean≈0.9, std≈0.8），不是 raw BOLD |
| EEG 数据归一化 | 已预处理（mean≈0, std≈0.01, max=1.0） |
| LR warmup 存在 | SAT 框架内置，lr 5e-6 → 9.8e-5 → cosine decay |

## 重训计划

**不需要从 Stage 1 重训**（reviewer 建议"从 Stage 1 重新训练"——我们不同意）：
- Stage 1 不计算 L_diff，动态缩放不触发
- LoRA alpha 不影响 Stage 1（Stage 1 不启用 LoRA）
- P1 的 temporal 结果有效（Valid L_temp_delta 下降 + 泛化）

**从 Stage 2 checkpoint 重跑 Stage 3**：
- 应用所有 6 个 fix
- 预期 DiT LoRA 的 scaling=1.0 会带来显著改善
- gradient_clipping=1.0 缓解梯度截断
- debug logging 提供实时诊断

**风险管理**：Fix 1 + Fix 4 叠加放大 LoRA 有效信号约 1280 倍。新训练前 100 iter 密切观察：
- 如果 loss spike / NaN → 降 lr 到 5e-5
- 如果 `debug/sf_diff_ratio` > 5 → 降 lambda_sf 到 0.001-0.002
- 如果 `debug/sf_diff_ratio` < 0.3 → 升 lambda_sf 到 0.005-0.01

## 延后项（后续版本）

| 项目 | 来源 | 原因 |
|------|------|------|
| Guidance cross-attention 注入 | Issue #4 | 方法改进，非 bug |
| GatedFusion 分别投影 | P1-9 | 已知技术债，架构变更 |
| Brain tokens 降到 64 | Issue #5 | 架构变更，需消融验证 |
| L_gm motion guidance loss | P1-6 | 缺少合适的 ground truth |
| Queue-based contrastive | P1-5 | 长期优化，已有替代方案 |
| Text caption 统一处理 | P0-5 | 正式实验前修复 |
| _clip_loss L2 归一化 | P0-6 | 启用 clip mode 前修复 |
