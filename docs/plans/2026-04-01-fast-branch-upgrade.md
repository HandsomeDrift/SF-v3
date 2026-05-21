# Fast Branch 全链路升级 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 重构 Fast Branch 的 target/encoder/head/loss/训练策略，使 L_fast 从基本不降（3%）变为与 L_slow 同量级的有效下降。

**Architecture:** 三层改进——(1) Target 重构：dynamics 改 3 分类、flow tokens PCA 降维 + log-zscore、新增运动方向分类、恢复 TC；(2) EEG Encoder 升级：Multi-Scale TCN 前端 + Temporal Attention Pooling；(3) Curriculum Training：4 阶段渐进训练。每层改进后都先跑验证实验，确认有效再叠加下一层。

**Tech Stack:** PyTorch, DeepSpeed, SAT, CogVideoX-5B

---

## 验证实验设计原则

**每个 Task 完成后必须通过以下验证才能继续：**

1. **Preflight Check**：跑 `tools/preflight_check.py`，11 项全 PASS
2. **Overfit Test**：用 1 个样本训 100 步，验证 loss 能降到接近 0（证明模型有能力学到 target）
3. **Mini Train**：用 200 步训练观察各项子 loss 趋势（证明大规模训练的方向正确）

---

## Phase A: Target 重构 + Head 改造

### Task 1: 提取新 supervision targets

**Files:**
- Create: `CineBrain/tools/extract_targets_v3.py`

**目的:** 从已有 shard 数据中计算新的 target，保存为增强版 shard。

**需要生成的新 target：**
- `dyn_class_3`: 3 分类标签 {0=slow, 1=mid, 2=fast}，基于 flow_mag 三分位数（p33=121.26, p66=200.87）
- `flow_token_pca`: PCA 降维后的 128-dim 向量（log(1+x) + z-score + PCA）
- `motion_dir_8`: 8 方向分类标签（从 flow_token 空间分布推导）
- `ofs_log_zscore`: log(1+ofs_score) z-score 归一化标量

**PCA 变换矩阵和统计量已保存在:**
`/public/home/maoyaoxin/zhangt/xxt/datasets/supervision_cache/version_v1/metadata/target_transform_stats.pt`

**脚本逻辑:**
```python
stats = torch.load("metadata/target_transform_stats.pt")
pca_components = stats["pca_components"]  # (128, 1920)
log_mean = stats["log_mean"]  # (1920,)
log_std = stats["log_std"]    # (1920,)

for each shard:
    # flow_token_pca
    log_ft = torch.log1p(flow_token)
    zscore_ft = (log_ft - log_mean) / (log_std + 1e-8)
    pca_ft = zscore_ft @ pca_components.T  # (N, 128)

    # dyn_class_3
    dyn_class = torch.where(flow_mag < 121.26, 0, torch.where(flow_mag < 200.87, 1, 2))

    # motion_dir_8
    ft_2d = flow_token.reshape(N, 32, 60)
    dx = ft_2d[:,:,30:].mean((1,2)) - ft_2d[:,:,:30].mean((1,2))
    dy = ft_2d[:,16:,:].mean((1,2)) - ft_2d[:,:16,:].mean((1,2))
    dir_label = ((torch.atan2(dy, dx) + pi) / (2*pi) * 8).long() % 8

    # ofs_log_zscore
    ofs_normed = (torch.log1p(ofs_score) - 3.9679) / (0.8708 + 1e-8)

    # Save back to shard
    shard["flow_token_pca"] = pca_ft
    shard["dyn_class_3"] = dyn_class
    shard["motion_dir_8"] = dir_label
    shard["ofs_log_zscore"] = ofs_normed
```

**验证:** 检查各 target 的分布是否合理（分类平衡、回归范围健康）

---

### Task 2: 更新 data_video.py 加载新 targets

**Files:**
- Modify: `CineBrain/data_video.py`

**改动：**
1. 在 v2_sharded 预加载的 `target_keys` 列表中添加新 key
2. 更新 key 映射
3. 移除旧的 L2 归一化和 flow_mag z-score（新 target 已预处理）
4. 移除 `del sf_targets["gt_tc_embed"]`（TC 恢复使用）

**新的 key 映射：**
```python
_name_map = {
    "keyframe_img_emb": "gt_keyframe_embed",
    "scene_text_emb": "gt_text_embed",
    "structure_latent": "gt_structure_embed",
    "flow_token_pca": "gt_motion_embed",     # 新: PCA 128-dim
    "dyn_class_3": "gt_dynamics_class",       # 新: 3 分类
    "motion_dir_8": "gt_direction_class",     # 新: 8 分类
    "ofs_log_zscore": "gt_tc_embed",          # 新: 归一化后的 OFS
}
```

**验证:** `preflight_check.py` 确认新 target 加载正确

---

### Task 3: 改造 Fast Branch heads

**Files:**
- Modify: `CineBrain/sgm/modules/encoders/fast_branch.py`

**改动：**

```python
class DynamicsHead(nn.Module):
    """3-class motion intensity classification: slow/mid/fast."""
    def __init__(self, in_dim=2048, num_classes=3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, num_classes),
        )
    def forward(self, x):
        return self.proj(x.mean(dim=1))  # (B, 3) logits

class MotionHead(nn.Module):
    """Predict PCA-reduced motion flow embedding."""
    def __init__(self, in_dim=2048, out_dim=128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )
    def forward(self, x):
        return self.proj(x.mean(dim=1))  # (B, 128)

class MotionDirectionHead(nn.Module):
    """8-class motion direction classification."""
    def __init__(self, in_dim=2048, num_classes=8):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, num_classes),
        )
    def forward(self, x):
        return self.proj(x.mean(dim=1))  # (B, 8) logits

class TemporalCoherenceHead(nn.Module):
    """Predict temporal coherence (OFS log-zscore, scalar regression)."""
    def __init__(self, in_dim=2048, out_dim=1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim // 4),
            nn.GELU(),
            nn.Linear(in_dim // 4, out_dim),
        )
    def forward(self, x):
        return self.proj(x.mean(dim=1)).squeeze(-1)  # (B,)
```

**FastBranch 添加 MotionDirectionHead，更新 forward 返回 z_dir。**

**验证:** Overfit test — 1 样本 100 步，L_dyn(CE) 和 L_mot(MSE+cos) 能降到接近 0

---

### Task 4: 更新 sf_losses.py

**Files:**
- Modify: `CineBrain/sgm/modules/diffusionmodules/sf_losses.py`

**改动：**

```python
class FastBranchLoss(nn.Module):
    def __init__(self, lambda_dyn=1.0, lambda_mot=1.0, lambda_tc=0.5, lambda_dir=0.5):
        super().__init__()
        self.lambda_dyn = lambda_dyn
        self.lambda_mot = lambda_mot
        self.lambda_tc = lambda_tc
        self.lambda_dir = lambda_dir

    def forward(self, fast_out, targets):
        _ref = next(iter(fast_out.values()))
        losses = {}
        total = _ref.new_tensor(0.0)

        # Dynamics: CrossEntropy (3-class)
        if "z_dyn" in fast_out and "gt_dynamics_class" in targets:
            logits = fast_out["z_dyn"]  # (B, 3)
            labels = targets["gt_dynamics_class"].long()  # (B,)
            losses["L_dyn"] = F.cross_entropy(logits, labels)
            total = total + self.lambda_dyn * losses["L_dyn"]

        # Motion: cosine + MSE hybrid (PCA 128-dim)
        if "z_mot" in fast_out and "gt_motion_embed" in targets:
            pred = fast_out["z_mot"]
            gt = targets["gt_motion_embed"]
            mse = F.mse_loss(pred, gt)
            cos = 1.0 - F.cosine_similarity(pred, gt, dim=-1).mean()
            losses["L_mot"] = 0.5 * mse + 0.5 * cos
            total = total + self.lambda_mot * losses["L_mot"]

        # Temporal coherence: SmoothL1 (scalar regression)
        if "z_tc" in fast_out and "gt_tc_embed" in targets:
            losses["L_tc"] = F.smooth_l1_loss(fast_out["z_tc"], targets["gt_tc_embed"])
            total = total + self.lambda_tc * losses["L_tc"]

        # Motion direction: CrossEntropy (8-class)
        if "z_dir" in fast_out and "gt_direction_class" in targets:
            logits = fast_out["z_dir"]  # (B, 8)
            labels = targets["gt_direction_class"].long()  # (B,)
            losses["L_dir"] = F.cross_entropy(logits, labels)
            total = total + self.lambda_dir * losses["L_dir"]

        return total, losses
```

---

### Task 5: 更新 MultiGuidanceAdapter 和 SFBrainEmbedder

**Files:**
- Modify: `CineBrain/sgm/modules/encoders/multi_guidance.py`
- Modify: `CineBrain/sgm/modules/encoders/sf_embedder.py`
- Modify: `CineBrain/configs/sf_v1/cinebrain_sf_v1_model.yaml`

**改动：**
- mot_input_dim: 1922 → 128+3+1+8=140（或直接用 z_mot 128-dim + z_dyn logits 3 + z_tc 1 + z_dir logits 8）
- 实际上 guidance 不需要用 logits，用 softmax 后的概率或直接用 embedding 更合理
- 简化方案：mot_input_dim = 128 (z_mot only)，分类结果通过其他方式注入

**验证:** Preflight + overfit test

---

### Task 6: 验证实验 — Phase A 效果确认

**验证步骤：**

1. **Preflight Check**: 全部 PASS
2. **Overfit Test (1 样本 100 步)**:
   - L_dyn (CE): 从 ~1.1 降到 < 0.1
   - L_mot (cos+MSE): 从 ~1.0 降到 < 0.1
   - L_tc (SmoothL1): 从 ~1.0 降到 < 0.1
   - L_dir (CE): 从 ~2.1 降到 < 0.1
3. **Mini Train (200 步)**:
   - 各子 loss 均有明显下降趋势
   - L_fast 总体下降 > 30%

**如果验证不通过，debug 后重试，不进入 Phase B。**

---

## Phase B: EEG Encoder 升级

### Task 7: 添加 Multi-Scale Temporal Convolution 模块

**Files:**
- Create: `CineBrain/sgm/modules/encoders/temporal_conv.py`

**新模块：**
```python
class MultiScaleTCN(nn.Module):
    """Multi-scale temporal convolution for EEG feature extraction."""
    def __init__(self, in_channels=2048, out_channels=2048, kernel_sizes=[3, 5, 9, 15]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels // len(kernel_sizes), k, padding=k//2),
                nn.BatchNorm1d(out_channels // len(kernel_sizes)),
                nn.GELU(),
            ) for k in kernel_sizes
        ])
        self.proj = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        # x: (B, S, D) → conv1d expects (B, D, S)
        x_t = x.transpose(1, 2)
        feats = [conv(x_t) for conv in self.convs]
        x_out = torch.cat(feats, dim=1).transpose(1, 2)  # (B, S, D)
        return self.proj(x_out)
```

### Task 8: 添加 Temporal Attention Pooling

**Files:**
- Modify: `CineBrain/sgm/modules/encoders/fast_branch.py`

**改动：** 替换各 head 中的 `x.mean(dim=1)` 为 temporal attention pooling：

```python
class TemporalAttentionPool(nn.Module):
    def __init__(self, dim=2048, num_heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        # x: (B, S, D)
        q = self.query.expand(x.shape[0], -1, -1)
        out, _ = self.attn(q, x, x)
        return out.squeeze(1)  # (B, D)
```

### Task 9: 集成到 EEG Encoder + FastBranch

**Files:**
- Modify: `CineBrain/sgm/modules/encoders/eeg_encoder_custom.py`
- Modify: `CineBrain/sgm/modules/encoders/fast_branch.py`

**改动：**
1. `CustomEEGTransformer` 中添加 `MultiScaleTCN` 在 Transformer 之前
2. `FastBranch` 中各 head 使用 `TemporalAttentionPool` 替代 mean pooling

**验证:** 同 Task 6 — overfit test + mini train，对比 Phase A 结果

---

## Phase C: Curriculum Training

### Task 10: 实现分阶段训练配置

**Files:**
- Create: `CineBrain/configs/sf_v1/sf_v1_train_stage1a.yaml` (Slow only)
- Create: `CineBrain/configs/sf_v1/sf_v1_train_stage1b.yaml` (Fast easy: classification only)
- Create: `CineBrain/configs/sf_v1/sf_v1_train_stage1c.yaml` (Fast full)
- Modify: `CineBrain/diffusion_video_brain.py` (支持冻结单个 branch)

### Task 11: 实现辅助 EEG-fMRI 对齐 loss

**Files:**
- Modify: `CineBrain/sgm/modules/diffusionmodules/sf_losses.py`

**新增：**
```python
class AuxAlignmentLoss(nn.Module):
    """InfoNCE: pull eeg_cls toward fmri_cls (detached)."""
    def forward(self, eeg_cls, fmri_cls):
        fmri_cls = fmri_cls.detach()  # stop gradient to slow branch
        # standard InfoNCE
        ...
```

### Task 12: 完整 4 阶段训练 + 效果验证

**训练计划：**
```
Stage 1-A: 3000 iter, Slow only
Stage 1-B: 3000 iter, Fast classification only (dyn + dir)
Stage 1-C: 3000 iter, Fast full (+ mot + tc)
Stage 2:   5000 iter, Joint + L_aux
```

**每阶段完成后检查 loss 趋势，确认方向正确再进入下一阶段。**

---

## 预估时间

| Phase | 编码 | 验证 | 训练 |
|-------|------|------|------|
| A (Target + Head) | 3h | 1h | - |
| A 验证实验 | - | - | 1h (overfit + mini) |
| B (EEG Encoder) | 2h | 1h | - |
| B 验证实验 | - | - | 1h |
| C (Curriculum) | 2h | 0.5h | 14h (4 stage × ~3.5h) |
| **Total** | **7h** | **3.5h** | **16h** |
