# Stage 2 Fusion Training — 设计文档

**Date:** 2026-04-03
**Context:** P1 评估 3/4 通过（Fast/Slow 正交 cosine≈0，temporal 可学习，flow_traj 边界，gating 微弱），准备进入 Stage 2 训练 GatedFusion + MultiGuidanceAdapter。

---

## 1. 目标

让 GatedFusion 和 MultiGuidanceAdapter 学习如何融合 P1 后**正交的 Fast/Slow 特征**，为 DiT 提供高质量的 context。

P1 训练后 Fast Branch 特征发生了根本性变化（cosine 从 ~0.9 降到 ~0），GatedFusion 的旧权重完全不适配新特征分布。Stage 2 的核心是让 fusion 层重新学习。

---

## 2. 关键设计决策

### 2.1 梯度问题与解决方案

**问题**: 经过代码分析确认，现有所有 loss（L_align, L_slow, L_fast, L_guide）的梯度都不经过 GatedFusion——它们直接监督 branch heads 的输出。如果不启用 L_diff，GatedFusion 是"死模块"。

**解决方案**: 在 Stage 2 启用 L_diff（diffusion denoising loss），冻结 DiT 但让梯度穿透。

**验证结果** (tools/verify_gradient_passthrough.py):
- 穿过 frozen proxy 后，GatedFusion 梯度 norm=3.21e-04（非零，有意义）
- 真实 42 层 DiT 衰减更多，但可通过调大 lr 补偿
- Branches 正确冻结（无梯度泄漏）

### 2.2 冻结策略

| 模块 | 状态 | requires_grad | 说明 |
|------|------|:---:|------|
| Slow Branch | 冻结 | False | Stage 1 已训好 |
| Fast Branch | 冻结 | False | P1 已验证 |
| fMRI/EEG Encoder | 冻结 | False | 同上 |
| **GatedFusion** | **解冻** | **True** | 学习融合正交特征 |
| **MultiGuidanceAdapter** | **解冻** | **True** | 学习 guidance 组合 |
| DiT | 冻结（梯度穿透） | False | 参数不更新，但梯度流过 cross-attention |
| First Stage (VAE) | 冻结 | False | 不变 |

**关键实现**: `embedder.is_trainable` 必须为 `True`（否则 GeneralConditioner 在 `torch.no_grad()` 下执行 embedder forward，梯度完全截断）。DiT 用 `param.requires_grad_(False)` 冻结（不阻止梯度穿透），不用 `torch.no_grad()`。

### 2.3 Loss 构成

```
L_total = L_diff                                    # 主：训练 fusion 的唯一信号
        + λ_slow * L_slow                            # 监控：保持 slow 监督（branch 冻结不更新）
        + λ_fast * (L_distill + L_temporal + ...)    # 监控：保持 fast 监督（branch 冻结不更新）
```

L_slow / L_fast 继续计算用于 loss monitoring，但因 branch 冻结不产生参数更新。

**λ 配置（Stage C）**:
```yaml
lambda_distill_cls: 0.2
lambda_distill_spatial: 0.2
lambda_temporal_delta: 1.0
lambda_temporal_abs: 0.2
lambda_flow_traj: 0.3
lambda_dyn: 0.1
```

---

## 3. 必要的代码改动

### 3.1 loss.py — 在 "fusion" 阶段启用 L_diff

当前 L282-283: `if self.training_stage == "joint":` 计算 L_diff。

改为: `if self.training_stage in ("fusion", "joint"):` 使 L_diff 在 fusion 阶段也生效。

### 3.2 训练配置 — 确保正确的冻结行为

```yaml
# configs/sf_v1/sf_v1_stage2_fusion.yaml
model:
  not_trainable_prefixes:
    - all              # 默认冻结所有
  # 然后通过 freeze_slow/fast_branch 控制 branch
  freeze_slow_branch: true
  freeze_fast_branch: true
  loss_fn_config:
    params:
      training_stage: fusion    # 启用 L_diff + L_guide
      sf_loss_config:
        lambda_distill_cls: 0.2
        lambda_distill_spatial: 0.2
        lambda_temporal_delta: 1.0
        lambda_temporal_abs: 0.2
        lambda_flow_traj: 0.3
        lambda_dyn: 0.1
```

**问题**: `not_trainable_prefixes: [all]` 会冻结所有参数，包括 GatedFusion。需要额外逻辑在冻结后**重新解冻**指定模块。

### 3.3 diffusion_video_brain.py — 新增 fusion 解冻逻辑

在现有的 `_freeze_slow` / `_freeze_fast` 之后，新增：
```python
# Stage 2: unfreeze fusion modules
if model_config.get('unfreeze_fusion', False):
    for n, p in self.named_parameters():
        if "gated_fusion" in n or "guidance_adapter" in n:
            p.requires_grad_(True)
```

### 3.4 conditioner config — embedder.is_trainable=True

当前 `is_trainable: true` 在 model yaml 的 conditioner 配置中。Stage 2 需确保这个值为 true。

### 3.5 DiT 冻结方式确认

当前 `not_trainable_prefixes: [all]` 已经把 DiT 设为 `requires_grad=False`。这不会阻止梯度穿透，只要 forward 不在 `no_grad` 下执行——而 `VideoDiffusionLossSF.__call__` 中 DiT forward 确实不在 `no_grad` 下。✓

---

## 4. 训练参数

| 参数 | 值 | 理由 |
|------|------|------|
| 起始 checkpoint | P1 v2 iter 3000 | 最新验证通过 |
| 迭代数 | 2000 | fusion 参数量小（~90 params），但梯度衰减需要多步 |
| 学习率 | 1e-4 | 梯度穿透 42 层 DiT 后衰减大，需要较高 lr |
| GPU | gpu2 卡3-7 (5卡) | 需 DiT forward/backward，显存需求大 |
| batch_size | 1 per GPU | 同 Stage 1 |
| gradient_accumulation | 2 | 同 Stage 1 |
| eval_interval | 500 | 监控 L_diff 下降和 gating 分化 |
| save_interval | 500 | 阶段较短，更频繁保存 |

---

## 5. 验证计划

### 5.1 Preflight（代码改动后）
- 跑 `tools/preflight_check.py` 确认 forward 正常
- 确认只有 GatedFusion + MultiGuidanceAdapter 有 `requires_grad=True`

### 5.2 Overfit test（1 sample, 200 steps）
- L_diff 应该能下降
- GatedFusion 参数 grad norm 非零且不 vanish

### 5.3 Mini train（mini500, 500 steps）
- L_diff 应稳定下降
- Gating weights 开始分化（高/低动态 clip 的 α_mot 差异变大）

### 5.4 完整训练后
- 重跑 `tools/evaluate_p1.py` 看 gating 行为是否改善
- L_diff 相比未训 fusion 的基线应该下降
- 可选：生成视频定性检查

---

## 6. 风险与缓解

| 风险 | 缓解 |
|------|------|
| DiT 42 层梯度消失 | overfit test 快速检测。如失败→加 auxiliary context loss 或用更高 lr |
| DiT forward/backward OOM | gradient checkpointing + bf16，与 Stage 1 配置一致 |
| GatedFusion 破坏 context 分布 | lr 不超过 1e-4 + 短训练 (2000 iter) |
| `not_trainable_prefixes: [all]` 与 fusion 解冻冲突 | 明确的解冻优先级：先冻结 all → 再解冻 gated_fusion + guidance_adapter |

---

## 7. 文献改进集成计划

以下改进**不在 Stage 2 实施**，留到 Stage 3（解冻 DiT + Fast Branch 时一起加入）：

| 改进 | 阶段 | 来源 |
|------|------|------|
| L_Struct（帧间结构相似性 loss） | Stage 3 | DynaMind |
| Causal mask（TemporalDynamicsDecoder） | Stage 3 | EEG2Video + MindCine |
| DANA 噪声初始化 | 推理时 | EEG2Video |
| α-Guidance SDEdit | 推理时 | NeuroClips |
