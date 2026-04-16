# Fast Branch 重设计：从光流分类到时序动态蒸馏

**Date:** 2026-04-02
**Revised:** 2026-04-02（根据 review 反馈修订 P1 方案）
**Context:** CineBrain-SF v1 的 Fast Branch（EEG 分支）经历了两次设计迭代。本文档完整记录遇到的问题、分析过程、以及两阶段解决方案的设计思路。

---

## 1. 原始设计回顾

### 1.1 Slow-Fast 假设

CineBrain-SF v1 的核心假设是**显式的 Slow-Fast 角色分配**优于统一的多模态融合：

- **Slow Branch (fMRI):** 负责语义和结构 — 关键帧内容、场景文本、空间布局
- **Fast Branch (EEG):** 负责运动和动态 — 运动速度、运动方向、时间一致性

这个假设基于两个模态的物理特性：fMRI 空间分辨率高但时间分辨率低（~2s/TR），EEG 时间分辨率高（1000Hz）但空间分辨率低。

### 1.2 原始 Fast Branch 架构

Fast Branch 原始设计包含 4 个预测头，分别对 RAFT 光流的不同统计量做分类/回归：

```
EEG (B, 5, 64, 800)
  ↓ EEG Encoder (12层 Transformer)
eeg_spatial (B, 226, 2048)
  ↓ TemporalAttentionPool
eeg_pooled (B, 2048)
  ├→ DynamicsHead:     3-class 分类 (slow/mid/fast)     ← target: dyn_class_3
  ├→ MotionHead:       128-dim 回归                      ← target: flow_token_pca
  ├→ DirectionHead:    8-class 分类 (运动方向)            ← target: motion_dir_8
  └→ TCHead:           标量回归 (时间一致性)              ← target: ofs_log_zscore
```

所有 supervision targets 来自 RAFT 光流模型对视频帧的计算：
- `dyn_class_3`: flow magnitude 的三分位数分类
- `flow_token_pca`: 光流向量的 PCA 降维表示
- `motion_dir_8`: 光流方向的 8 类离散化
- `ofs_log_zscore`: 帧间光流一致性分数

---

## 2. 遇到的问题

### 2.1 实验结果：L_fast 完全不下降

经过系统性实验验证（2026-04-01），在所有配置下 L_fast 均停留在随机水平：

| 实验配置 | 数据量 | 迭代数 | L_fast 结果 | 随机基线 |
|---------|--------|--------|------------|---------|
| 单样本 overfit | 1 | 200 | → 10⁻⁶ | — |
| mini + dyn+dir | 498 | 1500 | ≈ 2.14 | 2.14 |
| mini + dyn only | 498 | 1500 | train 1.05 / valid 1.21 | 1.10 |
| 完整数据集 | 4860 | 3000 | ≈ 2.10 | 2.14 |

**关键观察：**
- 单样本能 overfit → 模型架构和梯度链路没问题
- mini/完整数据集都卡在随机水平 → **EEG 信号本身无法预测这些 targets**
- 仅 dynamics 分类时有轻微过拟合但不泛化 → 即使最粗粒度的运动/静态分类也不可靠

### 2.2 根因分析

分析后确定了 4 个根本原因：

**① RAFT 光流与 EEG 感知存在语义鸿沟**

RAFT 计算的是**像素级运动**（包含相机运动、背景光流等），而 EEG 记录的是**大脑视觉皮层对运动的神经响应**。两者之间没有直接的映射关系。例如，一个缓慢平移的镜头在 RAFT 看来是全局大光流，但对视觉系统来说可能是"静态场景"。

**② EEG 空间分辨率不足以捕获 V5/MT+ 运动信号**

EEG 的 64 通道头皮电极对深层运动皮层（V5/MT+）的信号捕获能力有限。文献（EEG2Video, NeurIPS'24）也确认 EEG 能解码粗粒度动态程度（动态 vs 静态）但**无法解码细粒度光流方向或速度**。

**③ Slow clips 的方向标签是噪声**

对于几乎没有运动的 slow clips（约占训练数据的 1/3），`atan2` 计算出的方向角本质上是随机的（flow magnitude ≈ 0 时 arctan 无意义）。这些噪声标签会干扰方向分类的学习。

**④ L_align 因 batch_size=1 完全退化**

CineBrain 原始设计中重要的 EEG-fMRI 对齐 loss（InfoNCE）在 batch_size=1 时退化为常数 0（没有负样本）。这意味着 EEG encoder 从未接受过跨模态对齐的训练信号。

### 2.3 核心结论

> **EEG 信号无法泛化学习 RAFT 光流的统计量。** 原始 Fast Branch 的设计假设（EEG → 像素级运动特征）与 EEG 的物理能力不匹配。需要根本性重新设计。

---

## 3. 文献调研：EEG 到底能做什么？

在重新设计之前，我们调研了 2024-2025 年的相关工作，明确 EEG 在视频重建任务中的能力边界。

### 3.1 EEG 已验证可解码的信息

| 信息类型 | 文献 | 方法 |
|---------|------|------|
| 视觉语义类别 | EEG2Video | Seq2Seq 映射到语义嵌入空间 |
| 粗粒度动态程度 | EEG2Video | 动态 vs 静态二分类 |
| 场景切换/时序边界 | ERP 文献 | 事件相关电位 |
| 与 fMRI 对齐的表征 | CineBrain, MindEye2 | Contrastive learning |

### 3.2 EEG 不可解码的信息

| 信息类型 | 我们的实验 | 说明 |
|---------|-----------|------|
| 细粒度光流方向 (8-class) | L_fast ≈ random | 实验确认 |
| 细粒度运动速度 (3-class) | 过拟合不泛化 | 实验确认 |
| 人脸/人物检测 | EEG2Video 结论 | 文献报告 |

### 3.3 核心洞察

文献给出了一个一致的结论：**EEG 不应该直接做像素级/统计级的运动预测，而应该通过对齐到已有的高质量表征空间来发挥作用**。

几个关键设计范式：
1. **fMRI→EEG 蒸馏** (MindEye2): 用训好的 fMRI encoder 做 teacher，让 EEG encoder 学习对齐到同一表征空间
2. **时序密集对齐** (EEG2Video): 利用 EEG 的时间分辨率优势，做 Seq2Seq 映射到多帧潜变量序列
3. **功能分解** (DecoFuse): What/Where/How 分工，EEG 负责 "How"（时序动态如何变化）

---

## 4. P0 方案：fMRI→EEG 特征蒸馏

### 4.1 设计思路

既然 EEG 无法直接预测视频运动特征，那就让它**学习逼近 fMRI 的特征空间**。理由：

- Stage 1A 已经训练好了 fMRI encoder（Slow Branch），它的 spatial features `(B, 226, 2048)` 包含了丰富的视觉语义信息
- EEG 虽然分辨率不同，但与 fMRI 观察的是同一段视觉刺激，理论上应该能学到部分共享表征
- MSE 蒸馏不受 batch_size 限制（不像 InfoNCE 需要负样本），在 bs=1 下依然有效

### 4.2 具体改动

| 文件 | 改动 |
|------|------|
| `fast_branch.py` | 移除 4 个分类/回归 heads → `DistillationProjector` (LayerNorm + MLP) |
| `sf_losses.py` | `FastBranchLoss` (4 个 CE/MSE) → `FastBranchDistillLoss` (2 个 MSE) |
| `multi_guidance.py` | motion guidance 输入从 140-dim 分类输出 → 2048-dim 蒸馏特征 |

**蒸馏 Loss:**
```
L_distill_cls    = MSE(eeg_cls_proj,    fmri_cls.detach())      # CLS token 对齐 (1152-dim)
L_distill_spatial = MSE(eeg_pooled_proj, fmri_pooled.detach())   # 空间特征对齐 (2048-dim)
```

注意 `fmri_cls.detach()` — fMRI 特征作为固定 teacher，不接受梯度。

### 4.3 验证结果

| 实验 | L_fast 起始 | L_fast 最终 | 变化 |
|------|-----------|-----------|------|
| Overfit (1 sample) | 42.3 | **0.245** | -99.4% |
| Mini train (498) | 59.8 | **4.33** | -92.8% |
| Mini valid (498) | 55.3 | **1.70** | -96.9% |

**对比原始分类方案：** 分类版 L_fast 在 3000 iter 后仍在随机水平（2.10/2.14），蒸馏版仅 500 iter 就实现了 96.9% 的 validation loss 下降。

**结论：EEG encoder 完全能够通过蒸馏学习 fMRI 的特征表示，且泛化良好。**

### 4.4 P0 的定位与遗留问题

**P0 的核心价值不在于最终设计，而在于它是一个"能力探针"** — 它证明了 EEG branch 不是完全学不到东西，只是原始的光流监督错了。蒸馏 loss 大幅下降、且 validation 也跟着下降，足以说明 EEG encoder 本身和数据流没有根本性问题。

但 P0 引入了一个新的问题：

> **蒸馏后的 EEG 特征 ≈ fMRI 特征的压缩版。Fast Branch 没有提供 Slow Branch 无法提供的独有信息。**

具体表现：
- `eeg_pooled_proj (B, 2048)` ≈ `fmri_pooled (B, 2048)` （蒸馏目标就是让它们相等）
- GatedFusion 接收的 slow_feat 和 fast_feat 本质上是同一信息的两种精度版本
- EEG 的**核心优势——毫秒级时间分辨率——完全没有被利用**

这意味着：在 P0 的设计下，删掉 Fast Branch 对最终视频质量的影响可能微乎其微。**Slow-Fast 假设无法被验证。**

---

## 5. P1 方案：EEG 时序动态建模

### 5.1 设计思路

P0 证明了 EEG encoder 有学习能力，但学到的只是 fMRI 的"影子"。P1 的目标是让 EEG 提取 **fMRI 无法提供的时序动态信息**。

**关键洞察：**
- fMRI 的时间分辨率是 ~2 秒/TR，一段 4 秒视频只有 ~2-3 个有效时间采样点
- EEG 的时间分辨率是 1000Hz，同一段视频有 4000 个时间采样点
- 视频的时序动态（哪一帧变化大、场景何时切换、运动何时加速）是 EEG 能捕获但 fMRI 无法捕获的信息

**P1 的核心思想：让 EEG encoder 输出一个多帧的时序变化序列，重点监督"怎么变"而非"是什么"。**

这直接呼应了文献中的几个成功范式：
- EEG2Video 的 Seq2Seq 密集对齐
- NeuroClips 的 Perception Reconstructor（脑信号→"模糊视频序列"做时序锚）
- DecoFuse 的 "How" 因子（EEG 负责解码运动/时序动态如何变化）

### 5.2 新增模块：TemporalDynamicsDecoder

**位置：** 与 P0 的蒸馏路径并行，从同一个 EEG spatial features 分叉出来。

```
eeg_spatial (B, 226, 2048)      ← 来自 EEG Encoder
  │
  ├─ [P0 路径] TemporalAttentionPool → DistillationProjector
  │   → eeg_pooled_proj (B, 2048)      对齐到 fMRI 空间（保留）
  │
  └─ [P1 路径] TemporalDynamicsDecoder   ← 新增
      → temporal_dynamics (B, T, 1152)    多帧动态序列
```

**为什么是 cross-attention decoder？**

EEG encoder 输出的 226 个 tokens 混合了时间和空间信息（原始 800 个时间点被 pool 到 226 个 tokens）。我们不知道哪些 token 对应哪个时间段。Cross-attention 让 T 个时间 query 自由地从 226 个 EEG tokens 中**学习选择性聚合**，无需假设固定的时空映射。

**架构设计：**
- T_out 个可学习 temporal query tokens（T_out 从配置读取，需先确认 VAE 实际时间压缩维度，不写死）
- 1 个可学习 global dynamics query token（与 temporal queries 一起参与 cross-attention）
- 4 层 decoder，每层: cross-attention (query→EEG) + self-attention (query→query) + FFN
- D_dec = 512（bottleneck 设计，~10M 参数，相对主模型 2.3B 可忽略）
- 输出投影到 1152 维（匹配 SigLIP embedding 空间）

**双输出设计：**
```
TemporalDynamicsDecoder 输出:
  temporal_tokens:  (B, T, 1152)  — 逐帧时序特征，用于 delta/abs 监督 + 后续多帧引导
  global_dyn_token: (B, 1152)     — 全局动态摘要，用于 gated residual adapter + coarse dyn head
```

将 T_out+1 个 query 一起做 cross-attention，最后一个 query 的输出作为 `global_dyn_token`。两种输出的职责严格分离：

- **`temporal_tokens (B, T, 1152)`** — 仅用于逐帧时序监督（L_delta, L_abs, L_flow_traj），不直接参与 guidance
- **`global_dyn_token (B, 1152)`** — 仅用于 gated residual guidance adapter 和 coarse dynamics head，不参与逐帧监督

这让监督信号和推理路径各自独立，避免 guidance 路径的梯度干扰逐帧时序学习。

### 5.3 Supervision Target：Delta-Based 多帧时序监督

**原始素材：** 对每个视频 clip 的 33 帧均匀采样 T_out 帧，分别通过 SigLIP image encoder 获取 1152-dim embedding，得到 `temporal_frame_embs: (T, 1152)`。

**核心改进（基于 review 反馈）：不直接拟合绝对帧 embedding，而是以"变化量"为主监督。**

如果直接拟合 T 帧的绝对 SigLIP embedding，decoder 最容易学到的是静态语义信息（场景里有什么、整体外观），而不是动态信息（场景怎么变化）。这会让 Fast Branch 从"fMRI 的影子"变成"视频语义序列的影子"，仍然没有学到真正的动态优势。

**Delta supervision 设计：**

```
原始帧特征:  z_1, z_2, ..., z_T     (T, 1152) — SigLIP 绝对 embedding
Delta 特征:  Δz_t = z_t - z_1        (T, 1152) — 相对首帧的变化
```

**Loss 构成（双监督，非 fallback）：**
```
L_temporal = λ_delta * MSE(predicted_delta, gt_delta)          # 主：时序变化（delta-SigLIP）
           + λ_abs * MSE(predicted_abs, gt_abs)                # 辅：绝对定位（降权）
           + λ_flow_traj * MSE(predicted_flow_traj, gt_flow_traj)  # 辅：粗粒度运动轨迹锚点
```

其中 `λ_delta > λ_abs, λ_flow_traj`（建议 λ_delta=1.0, λ_abs=0.2, λ_flow_traj=0.3）。

**Coarse flow temporal summary（辅助动态锚点）：**

Delta-SigLIP 本质上仍来自静态图像 encoder 的帧级差分，不是专门为 motion 设计的。为增强"动态纯度"，从第一版开始就加入一个轻量的 flow-derived temporal summary 作为辅助监督：

- 已有 `flow_mag` (每 clip 一个标量) → 改为提取 **per-frame flow magnitude trajectory**: `(T,)` 向量
- 或用已有的 `flow_token (1920-dim)` 在时间维切分，得到粗粒度运动强度序列
- TemporalDynamicsDecoder 额外输出一个 `flow_traj_pred (B, T)` 标量序列，与 gt 对齐

这不会回到原来"重监督 RAFT 细粒度统计"的死路（只是粗粒度运动强度标量序列），但能明显增强 decoder 的"动态感知"。

**为什么用 Δz_t = z_t - z_1 而非 z_t - z_{t-1}？**
- 相对首帧的 delta 保留了全局变化轨迹，不丢累积信息
- 帧间差分 z_t - z_{t-1} 太局部，连续帧可能差异极小导致信号被噪声主导
- 方案 A (z_t - z_1) 作为初始选择，如果效果不够可切换到方案 B (z_t - z_{t-1}) 或混合

**Fallback 顺序（如果 delta-SigLIP 差异仍太小）：**
1. 先试 absolute + delta 混合监督
2. 如果差异太小，改为 only delta
3. 如果 still trivial，改为 delta cosine trajectory 或 pairwise temporal ranking
4. 最终 fallback：flow-derived coarse temporal tokens

**存储：** 提取时直接保存绝对帧特征 `temporal_frame_embs (T, 1152)`，delta 在训练时在线计算（无额外存储成本）。

### 5.4 保留轻量 Coarse Dynamics 辅助头

虽然细粒度光流分类失败了，但文献和我们自己的实验都表明 EEG 对**最粗粒度的动态程度**有一定信息（dynamics 分类"有轻微过拟合"）。完全丢弃这个信号是浪费。

**方案：** 保留一个极轻量的 dynamics 辅助头：
- 2-class 分类（static/dynamic），基于 flow_mag 中位数二分
- 权重很小（λ_dyn=0.1），仅作 regularizer
- 作用：(1) 给 EEG encoder 一个明确的"快慢"监督锚点；(2) 消融时更容易证明 Fast Branch 学到了"动态程度"

### 5.5 P0 蒸馏权重的训练调度

P0 蒸馏解决了"能不能学"的问题，但如果蒸馏权重始终很大，它会持续把 EEG features 往 "更像 fMRI" 方向拉，压制 P1 temporal decoder 学到的独有动态信息。

**训练调度设计：**

| 阶段 | 内容 | λ_distill | λ_temporal | λ_dyn |
|------|------|-----------|-----------|-------|
| Stage A (P0) | 纯蒸馏，建立基础表征 | 1.0 | 0 | 0 |
| Stage B (P1 warm-up) | 冻结/半冻结 P0 projector，训 temporal decoder | 0.5 | 1.0 | 0.1 |
| Stage C (Joint finetune) | 联合训练，蒸馏降权 | 0.2 | 1.0 | 0.1 |

核心原则：**训练后期 λ_temporal > λ_distill**，否则 P1 学到的独有时序信息会被 P0 蒸馏拉回 fMRI 空间。

### 5.6 Guidance 升级：先轻量后深入

**第一版（保守）：** 不深改 DiT context flow，而是用 gated residual adapter 增强现有 motion guidance 路径。直接使用 `global_dyn_token`。

```python
# P1 global dynamics token → gated residual adapter
g_temporal = self.temporal_proj(global_dyn_token)       # (B, 4096)
g_mot = g_mot + gate * g_temporal.unsqueeze(1)          # gated residual
```

理由：
- 更稳定，显存压力更小，容易做回退
- 更容易判断提升来自 P1 还是结构变化
- 等验证有效后，再升级为 multi-frame cross-attention 注入

**第二版（后续）：** cross-attention，context tokens 作为 query，temporal_dynamics 作为 key/value，每个 spatial token 从 T 帧中选择性聚合。

### 5.7 P0 + P1 的完整 Loss

```
L_fast = λ_distill * (L_distill_cls + L_distill_spatial)     # P0: fMRI 对齐
       + λ_temporal * (λ_delta * L_delta + λ_abs * L_abs)     # P1: 时序动态 (framewise)
       + λ_flow_traj * L_flow_traj                             # P1: 粗粒度运动轨迹
       + λ_dyn * L_dynamics                                    # 辅助: 粗粒度动态分类
```

---

## 6. 数据流全景 (P0 + P1)

```
EEG (B, 5, 64, 800)
  ↓ [EEG Encoder: Conv1d + TCN + 12层 Transformer]
eeg_cls (B, 1152)          eeg_spatial (B, 226, 2048)
                             │                    │
                    [TemporalAttentionPool]  [TemporalDynamicsDecoder (P1)]
                    (B, 2048)                (B, T, 1152)
                       │                        │
               [DistillationProjector]     ┌────┴─────┐
               eeg_pooled_proj (B,2048)    │ Delta    │ Abs (辅)
                    │                      │ target   │ target
                    │                      └────┬─────┘
                    │                    [CoarseDynHead]
             L_distill (P0)              L_temporal + L_dyn
                    │                           │
               ┌────┴────┐              ┌───────┴────────┐
               │ fMRI 对齐 │              │ 时序动态引导      │
               └────┬────┘              │ (gated adapter) │
                    │                   └───────┬────────┘
                    ↓                           ↓
          GatedFusion ───────────── MultiGuidanceAdapter
                    ↓
          context (B, 226, 4096) → DiT
```

---

## 7. 验证计划

### 7.1 标准验证（每个 step 后）

1. **Overfit test**: 1 sample, 200 steps — L_temporal_delta 应降到接近 0
2. **Mini train**: 498 samples, 500 steps — L_temporal_delta 应稳定下降，valid 也下降

### 7.2 P1 专属验证（review 建议新增）

**① Temporal target 可学习性检查：**
- 输出 predicted vs GT temporal sequence 的可视化
- 逐帧 cosine similarity heatmap（predicted frame i vs GT frame j）
- Delta embedding norm 曲线（高动态 clip 应有更大 delta norm）
- `predicted_flow_traj` vs `gt_flow_traj` 的定量指标：Pearson 相关系数、逐帧 MAE、高/低动态 clip 子集的分类 AUC

**② Fast Branch 互补性验证（消融实验）：**
- Slow only（基线）
- Slow + P0（蒸馏 Fast Branch）
- Slow + P0 + P1（完整方案）

比较维度：视频时序指标、场景切换片段质量、motion-sensitive 子集表现

**③ Gating 行为趋势性分析：**

分析 gating weights 与 clip 动态程度的关系，作为**趋势性证据**，不作为唯一成败判据：
- `alpha_mot` 在高动态 clip 中是否**趋势性**更高（箱线图 + Spearman 相关系数）
- `alpha_key/alpha_txt` 在静态 clip 中是否**趋势性**更高
- 预期：在早期训练中 gating 差异可能不显著，随训练推进应逐渐分化。如果 3000 iter 后仍完全无趋势，需检查 gating network 是否有效，但不应仅凭此否定整体方案

**④ Fast/Slow 特征独立性：**
- Fast 和 Slow 输出的 cosine similarity 分布
- 如果 P1 成功，Fast features 应与 Slow features 有显著差异（不完全塌缩）

---

## 8. 风险评估与缓解

### 8.1 EEG 226 tokens 中的时序信息可能不足

EEG encoder 在 `AdaptiveAvgPool1d(226)` 步骤中将 800 个时间点压缩为 226 个 tokens。如果这一步丢失了太多时序信息，TemporalDynamicsDecoder 可能无法恢复。

**缓解:** Overfit test 会快速验证。如果失败，可考虑修改 EEG encoder 内部结构——在 pool 之前保留 5 段的时序结构（5 个独立 token groups），让 decoder 能区分不同时间段。

### 8.2 相邻帧 SigLIP delta 可能过小

如果视频变化缓慢，delta embeddings 的 norm 可能极小，信噪比低。

**缓解（明确的 fallback 链）:**
1. 先试 absolute + delta 混合监督
2. 如果差异太小 → only delta
3. 如果 still trivial → delta cosine trajectory 或 pairwise temporal ranking
4. 最终 fallback → flow-derived coarse temporal tokens

### 8.3 P0 蒸馏压制 P1 独特性

蒸馏 loss 把 EEG features 往 fMRI 方向拉，可能压制 temporal decoder 的独有信息。

**缓解:** λ_distill staged decay（见 §5.5），训练后期 λ_temporal > λ_distill。

### 8.4 Temporal decoder 学到 trivial 解

如果所有帧的 delta 被预测为 0（平均值解），decoder 没有真正建模时序。

**缓解:** 检查 delta prediction 的方差。如果方差接近 0，加入 temporal diversity loss 或帧间对比 loss 强制不同帧预测有差异。

---

## 9. 实施步骤

### 前置步骤：确认 VAE 时间维度

在实现前必须确认：
1. 当前训练视频 clip 有多少帧 → 33
2. 经过 CogVideoX VAE 后时间维压缩成多少 → 需要实际验证
3. `num_temporal_queries` 设为 config 参数，根据上述结果设定

### 实施

| Step | 内容 | 依赖 |
|------|------|------|
| 0 | 确认 VAE 时间压缩维度，确定 T_out | 无 |
| 1 | 离线提取并缓存：`temporal_frame_embs (T, 1152)` — 多帧 SigLIP embeddings；`flow_mag_traj (T,)` — 逐帧 flow magnitude 轨迹；同时记录 `frame_sampling_rule`（采样帧 indices）和 `T_out` 到 shard metadata | Step 0 |
| 1b | 统计 delta 分布的**可分性**（不仅看 norm，还要看：同 clip 内不同时间点 delta 方差、高/低动态 clip 的 delta 分布差异、delta 与 flow_mag/scene_cut 的相关性） | Step 1 |
| 2 | 实现 `TemporalDynamicsDecoder` 模块 | 无 |
| 3 | 更新 `FastBranch` 集成 P1 decoder + coarse dyn head | Step 2 |
| 4 | 更新 `FastBranchDistillLoss` 加入 L_temporal (delta+abs) + L_dyn | Step 3 |
| 5 | 更新 `MultiGuidanceAdapter` — 轻量 gated residual adapter | Step 3 |
| 6 | 更新 data pipeline 和配置文件（T_out 可配） | Step 1 |
| 7 | Overfit test + Mini train + 可学习性检查 | Step 1-6 |
| 8 | 完整数据集 P1 训练（带 staged λ decay） | Step 7 通过 |

---

## 10. 预期结果

如果 P1 成功：
- EEG Fast Branch 不再是 fMRI 的影子，而是提供**独有的时序变化信息**
- Delta supervision 确保 decoder 学到的是"场景怎么变"而非"场景是什么"
- GatedFusion 接收到两种**真正互补的**模态信息：fMRI 的语义/结构 + EEG 的时序动态
- 生成的视频应在**运动流畅性和场景转换时序**上优于仅用 fMRI 的版本
- **Slow-Fast 假设可以被公正验证**：消融 Fast Branch 应导致运动质量显著下降，且 gating weights 应体现出 clip 类型相关的模态选择行为

---

## 11. 后续路线备忘

**Contrastive alignment 恢复计划：** P0 用 MSE 蒸馏绕开了 bs=1 下 InfoNCE 退化的问题，这在当前阶段合理。但后续如果需要恢复跨模态 contrastive alignment 以增强 Fast/Slow 互补性，将采用以下方案之一（而非依赖原始 batch negatives）：
- Memory bank / queue-based contrastive（MoCo 风格，维护历史 embedding 队列作为负样本）
- Gradient accumulation 后的 pseudo-batch negatives（累积多步的 embeddings 组成 virtual batch）
- Cross-sample within-subject negatives（同一被试不同 clip 的 embeddings 互为负样本）

这是一个已识别但暂未解决的问题，不影响 P1 的实施，但需要在 Stage 2 (Fusion) 之前决定是否启用。
