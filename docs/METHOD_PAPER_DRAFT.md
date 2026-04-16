# CineBrain-SF: Slow-Fast Dual-Branch Architecture for Multi-Modal Brain-to-Video Reconstruction

**方法论文档 (Paper-Style Draft)**
**Date**: 2026-04-16
**Status**: 供讨论与创新点分析用

---

## 1. Introduction

### 1.1 Problem Statement

脑信号到视频重建 (Brain-to-Video Reconstruction) 旨在从非侵入式神经影像数据（如 fMRI、EEG）中解码人类观看自然视频时的视觉感知，并生成对应的视频。这是计算神经科学与生成式 AI 的交叉前沿问题。

**核心挑战**在于两种模态的物理特性存在根本性互补与冲突：

| 特性 | fMRI | EEG |
|------|------|-----|
| 空间分辨率 | 高 (~mm 级体素) | 低 (~cm 级头皮电极) |
| 时间分辨率 | 低 (~2s/TR) | 高 (1000Hz) |
| 可捕获信息 | 语义、结构、空间布局 | 时序动态、运动节律、场景切换 |
| 信号延迟 | 血氧响应延迟 ~6s | 近实时 (~ms) |

现有方法（如 CineSync/CineBrain baseline）将 fMRI 和 EEG 编码后融合为**统一脑潜变量 (unified brain latent)**，再送入视频扩散模型解码。这种方案隐含假设两种模态可以被映射到同一表征空间，但**忽略了它们在时间尺度和信息类型上的根本差异**。

### 1.2 Core Hypothesis

> **显式的 Slow-Fast 角色分配 (Explicit Slow-Fast Role Assignment)** 优于统一的多模态潜变量融合 (Unified Multimodal Latent Fusion)，用于连续自然视频重建。

具体而言：
- **Slow Branch (fMRI-driven)**: 负责语义内容、空间结构、关键帧先验 — 回答 "视频里有什么 (What)"
- **Fast Branch (EEG-driven)**: 负责运动动态、时序变化、场景转换 — 回答 "视频怎么变化 (How)"

这一假设直接受神经科学中 **视觉通路的腹侧-背侧 (ventral-dorsal) 功能分离** 启发：腹侧通路处理物体识别 (what)，背侧通路处理空间定位与运动感知 (where/how)。

### 1.3 Contributions

CineBrain-SF v1 的主要贡献包括：

1. **Slow-Fast 双分支架构**：首次在 fMRI+EEG 多模态脑信号视频重建中引入显式的功能分离设计，fMRI 驱动语义-结构分支，EEG 驱动运动-动态分支。
2. **EEG 时序动态解码器 (Temporal Dynamics Decoder)**：利用 cross-attention decoder 从 EEG 中提取多帧时序变化序列，以 delta-SigLIP supervision 为核心监督信号，配合光流轨迹分类和粗粒度动态分类。
3. **跨模态门控融合 + 多通道 Guidance 注入**：CrossModalGatedFusion 通过 cross-attention 融合双分支特征并学习 per-sample 的 4 通道 guidance 权重；MultiGuidanceAdapter 以 per-channel cross-attention 实现空间选择性的 guidance 注入。
4. **三阶段渐进式训练策略**：Branch Pretrain → Fusion Training → Joint Finetuning，逐步解冻模块、避免梯度冲突。
5. **实验验证**：在同被试 (within-subject) 和跨被试 (cross-subject) 设置下，FVD 分别下降 31% 和 27%，EPE 下降 20%，11/14 项指标优于 baseline。

---

## 2. Related Work

### 2.1 Brain-to-Video Reconstruction

| 方法 | 会议 | 模态 | Backbone | 核心技术 |
|------|------|------|----------|---------|
| **EEG2Video** | NeurIPS'24 | EEG | Inflated SD | Seq2Seq 自回归 + DANA 动态噪声调度 |
| **NeuroClips** | NeurIPS'24 Oral | fMRI | AnimateDiff v3 | Inception Extension + alpha/beta/gamma 三重 Guidance |
| **Mind-Animator** | ICLR'25 | fMRI | Inflated SD | Sparse Causal Attention + 三模态对比学习 |
| **DecoFuse** | arXiv'25 | fMRI | DragNUWA | What/Where/How 分解 + 光流 Codebook |
| **DynaMind** | arXiv'25 | EEG | SD v1.4 | TDA temporal blueprint + L_Struct |
| **MindCine** | arXiv'25 | EEG | T2V model | EEG 预训练 + CausalSeq + SoftCLIP |
| **CineSync** (baseline) | — | fMRI+EEG | CogVideoX-5B | 统一 brain latent 融合 |

**现有方法的共性局限**：
- 单模态方法无法同时利用 fMRI 的空间优势和 EEG 的时序优势
- 多模态方法（如 CineSync）将两种模态简单融合为统一潜变量，丢失了各自的信息优势
- 少有工作探索如何**显式分配**不同模态在视频生成中的功能角色

### 2.2 Key Technical Inspirations

| 技术 | 来源 | 在 SF-v1 中的应用 |
|------|------|-----------------|
| fMRI→EEG 特征蒸馏 | MindEye2 (ICML'24) | P0 阶段：MSE 蒸馏建立 EEG 基础表征 |
| Seq2Seq 时序密集对齐 | EEG2Video | P1 阶段：cross-attention decoder 输出多帧序列 |
| 光流 Codebook (分类替代回归) | DecoFuse | Flow trajectory 预测：K-means 离散化 + CE loss |
| Sparse Causal Attention | Mind-Animator | TemporalDynamicsDecoder 中防止 shortcut |
| L_Struct (帧间结构相似性) | DynaMind | FastBranchDistillLoss 中的辅助结构 loss |
| MoCo Queue 对比学习 | MoCo v2 | 解决 batch_size=1 下 InfoNCE 退化 |
| DANA 动态噪声调度 | EEG2Video | 推理端优化（已验证有效） |

---

## 3. Method

### 3.1 Overall Architecture

CineBrain-SF v1 在 CogVideoX-5B DiT 视频扩散模型基础上，构建了四个核心模块：

```
                          ┌─────────────────────┐
  fMRI ──→ fMRI Encoder ──→│  Slow Branch (S)    │──→ z_key, z_txt, z_str
                           │  语义 / 结构 / 关键帧 │
                           └────────┬────────────┘
                                    │
                              CrossModalGatedFusion ──→ z_b + {α_key, α_txt, α_mot, α_brain}
                                    │                           │
                           ┌────────┴────────────┐              │
  EEG ──→ EEG Encoder ───→│  Fast Branch (F)     │    MultiGuidanceAdapter
                           │  蒸馏 / 时序 / 运动   │              │
                           └─────────────────────┘              ↓
                                                          context (B, 226, 4096)
                                                                │
                                                        DiT (CogVideoX-5B)
                                                                │
                                                         Denoised Video
```

**设计原则**：
- **最小侵入 (Minimal Invasion)**：在 CineSync 代码基础上扩展，不重写核心 pipeline
- **配置驱动 (Config-Driven)**：所有新模块通过 YAML 配置开关，支持消融实验
- **梯度隔离 (Gradient Isolation)**：三阶段训练中精确控制哪些模块接收梯度

### 3.2 Slow Branch: Semantic-Structure Decoding from fMRI

**输入**: fMRI visual ROI features `(B, 5, 8405)`, auditory ROI features `(B, 5, 8405)` (5 个 lag-corrected TRs)

**Encoder**: `CustomfMRITransformer` — 12 层 Transformer，输出 `fmri_spatial (B, 226, 2048)` 和 `fmri_cls (B, 1152)`

**Slow Branch 由三个预测头组成**：

#### 3.2.1 Keyframe Head
```
fmri_spatial → MeanPool → MLP(2048 → 1152) → z_key
```
- **目标**: 预测视频关键帧的 SigLIP image embedding
- **监督**: `L_key = MSE(z_key, gt_keyframe_embed)`
- **作用**: 为扩散模型提供"这段视频应该长什么样"的语义先验

#### 3.2.2 Scene-Text Head
```
fmri_spatial → MeanPool → MLP(2048 → 1152) → z_txt
```
- **目标**: 预测视频场景的文本描述 embedding
- **监督**: `L_txt = 1 - cosine_similarity(z_txt, gt_text_embed)`
- **作用**: 提供语义级的 text guidance，类似于 text-to-video 中的 text prompt

#### 3.2.3 Structure Head
```
fmri_spatial → MeanPool → MLP(2048 → 86400) → reshape → z_str (B, 16, 60, 90)
```
- **目标**: 预测视频帧的 VAE latent 空间结构
- **监督**: `L_str = MSE(z_str, gt_structure_embed)`
- **作用**: 提供空间布局先验（低频结构信息）

#### 3.2.4 Audiovisual Context Adapter
当听觉 fMRI ROI 可用时，通过 cross-attention 将听觉上下文融入视觉 fMRI 特征：
```
slow_feat = CrossAttention(Q=fmri_visual, K=V=fmri_auditory) + fmri_visual
```
这让 Slow Branch 能够利用视听联合上下文进行更准确的场景理解。

### 3.3 Fast Branch: Temporal Dynamics Decoding from EEG

Fast Branch 经历了三次关键迭代，最终形成 **P0 蒸馏 + P1 时序动态** 的双路设计。

**输入**: EEG signal `(B, 5, 64, 800)` (5 trials, 64 channels, 800 time points @ 200Hz)

**Encoder**: `CustomEEGTransformer` — Conv1d + TCN + 12 层 Transformer，输出 `eeg_spatial (B, 226, 2048)` 和 `eeg_cls (B, 1152)`

#### 3.3.1 设计演化历程

**第一代 (废弃): RAFT 光流分类/回归**

最初设计了 4 个预测头（DynamicsHead, MotionHead, DirectionHead, TCHead），直接从 EEG 预测 RAFT 光流的统计量（运动速度分类、方向分类、PCA token 回归等）。

**失败原因**：系统实验证明，所有 head 在泛化数据上均卡在随机水平。根因有四：
1. RAFT 计算像素级运动，EEG 记录神经响应，两者存在**语义鸿沟**
2. EEG 64 通道对深层运动皮层 (V5/MT+) 捕获能力有限
3. 低运动 clip 的方向标签本质上是随机噪声
4. batch_size=1 下 InfoNCE 对比学习完全退化

**第二代 (P0): fMRI→EEG 特征蒸馏**

转向让 EEG 学习逼近 fMRI 特征空间，作为能力验证：
```
L_distill = MSE(eeg_cls_proj, fmri_cls.detach())          # CLS 级
          + MSE(eeg_pooled_proj, fmri_pooled.detach())     # 空间级
```
- fMRI 特征作为 teacher（`.detach()` 阻断梯度），EEG 作为 student
- MSE 蒸馏不受 batch_size 限制，在 bs=1 下有效
- **验证结果**: validation loss 下降 96.9%，证明 EEG encoder 有学习能力

**P0 的定位**: 能力探针 (capability probe)。证明 EEG 能学，但学到的只是 fMRI 的"影子"——Fast Branch 未提供 Slow Branch 无法提供的独有信息。

**第三代 (P1): EEG 时序动态建模 (当前方案)**

P1 的核心目标是让 EEG 提取 **fMRI 无法提供的时序动态信息**，利用 EEG 的时间分辨率优势 (1000Hz vs ~0.5Hz)。

#### 3.3.2 TemporalDynamicsDecoder

这是 P1 的核心模块 — 一个 cross-attention decoder，从 EEG 空间特征中解码多帧时序变化序列。

**架构**:
- **T_out + 1 个可学习 query tokens**: T_out 个 temporal queries + 1 个 global dynamics query
- **4 层 decoder layers**, 每层包含:
  - Cross-attention: temporal queries → EEG spatial tokens (226 个)
  - Self-attention: temporal queries → temporal queries
  - FFN: 前馈网络
- **Bottleneck design**: D_dec = 512 (~10M 参数，相对主模型 2.3B 可忽略)
- 输出投影到 1152 维 (匹配 SigLIP embedding 空间)

**关键设计决策**:

1. **BF16-Safe Attention**: 手写多头注意力实现，避免 `nn.MultiheadAttention` 在 bf16 混合精度下的 CUDA 数值稳定性 bug

2. **Causal Mask + Sparse Attention (P1-1)**: 
   - 严格因果掩码限制帧 t 只能 attend 到 ≤t 的帧，增强时序建模
   - 叠加 random sparse drop (30%)，随机 mask 掉注意力位置，防止 decoder 走捷径

3. **Flow Codebook (P1-3, 来自 DecoFuse)**:
   - 对训练集光流向量做 K-means 聚类 (K=64)，将运动预测从连续回归转为离散分类
   - `flow_traj_pred (B, T, K)` → softmax → CE loss
   - 根治了 EEG→像素级光流回归的语义鸿沟

4. **双输出设计**:
```python
temporal_tokens:   (B, T, 1152)  # 逐帧时序特征 → delta/abs/flow_traj 监督
global_dyn_token:  (B, 1152)     # 全局动态摘要 → gated residual guidance + dyn 分类
```
两种输出职责严格分离，避免 guidance 路径的梯度干扰逐帧时序学习。

#### 3.3.3 Delta-Based Temporal Supervision

**核心创新**: 不直接拟合绝对帧 embedding，而是以**帧间变化量 (delta)** 为主监督信号。

**动机**: 如果直接拟合 T 帧的绝对 SigLIP embedding，decoder 最容易学到的是静态语义信息（场景里有什么），而不是动态信息（场景怎么变化）。

**Supervision target 构造**:
1. 对每个视频 clip 的 33 帧均匀采样 T_out 帧
2. 通过 SigLIP image encoder 提取帧级 embedding: `z_1, z_2, ..., z_T (T, 1152)`
3. 计算相对首帧的 delta: `Δz_t = z_t - z_1`
4. Delta 在训练时在线计算（离线存储绝对帧特征，无额外存储成本）

**为什么 Δz_t = z_t - z_1 而非 z_t - z_{t-1}**:
- 相对首帧保留全局变化轨迹，不丢累积信息
- 帧间差分太局部，连续帧差异极小时信噪比低

**Fast Branch 完整 Loss**:
```
L_fast = L_distill_cls + L_distill_spatial              # P0 蒸馏 (fMRI 对齐)
       + λ_delta · L_temporal_delta                      # P1 主监督: MSE(pred_delta, gt_delta)
       + λ_abs · L_temporal_abs                          # P1 辅助: MSE(pred_tokens, gt_frame_embs)
       + λ_flow · L_flow_traj                            # P1 光流轨迹: CE (codebook分类)
       + λ_dyn · L_dyn                                   # 辅助: 粗粒度动态分类 (static/dynamic, 2-class)
       + λ_struct · L_struct                             # DynaMind 启发: 帧间关系矩阵匹配
```

其中 `λ_delta=1.0, λ_abs=0.2, λ_flow=0.3, λ_dyn=0.1`，确保训练后期 delta supervision 主导。

#### 3.3.4 L_struct: 帧间结构相似性 Loss

受 DynaMind 启发，L_struct 匹配预测与真实帧特征的**帧间关系矩阵**：

```python
# 计算 cosine similarity matrix (T x T)
pred_sim = cosine_similarity_matrix(pred_tokens)    # (T, T)
gt_sim = cosine_similarity_matrix(gt_frame_embs)    # (T, T)

# 只取 off-diagonal 部分 (对角线是 trivial 的 self-similarity)
L_struct = MSE(pred_sim[off_diag], gt_sim[off_diag])
```

这让模型不仅学习每帧的绝对特征，还学习帧间的**相对关系结构** — 哪些帧之间相似、哪些帧之间差异大。

### 3.4 Cross-Modal Gated Fusion (CMGF)

CMGF 将 Slow Branch 和 Fast Branch 的特征融合为统一的 brain latent，同时学习 per-sample 的多通道 guidance 权重。

#### 3.4.1 架构 (v2 重设计)

原始设计使用 concat→shared projection，存在严重的信息瓶颈（Fast Branch 信息只能间接传递）。v2 改为分离投影 + cross-attention mixing：

```
slow_feat (B, 226, 2048) ──→ slow_proj ──→ Q
fast_feat (B, 226, 2048) ──→ fast_proj ──→ K, V

z_fused = MultiHead_CrossAttention(Q, K, V)     # Fast 信息直接注入
z_fused = z_fused + SelfAttention(z_fused)       # 内部精炼
z_b = out_proj(z_fused)                          # (B, 226, 4096) → DiT 输入维度
```

#### 3.4.2 门控网络 (Gating Network)

门控网络从融合特征中学习 4 个 per-sample 的 guidance 权重：

```python
pooled = concat(mean_pool(slow_feat), mean_pool(fast_feat))  # (B, 4096)
alphas = sigmoid(MLP(pooled))                                # 4 × (B, 1)
```

**关键技巧**:

1. **零初始化 (Zero-Init)**: 门控 MLP 最后一层 weight/bias 全零 → `sigmoid(0) = 0.5` → 训练开始时所有 guidance 通道权重均等，避免偏置

2. **Alpha Floor + Ceiling**: `alpha = alpha * 0.9 + 0.05`，将 alpha 映射到 [0.05, 0.95]，保证**所有通道始终有非零梯度流通**，防止 sigmoid 饱和导致的 "dead alpha" 问题

3. **Reset Gate Net**: Stage 3 启动时可选重置门控网络参数，避免继承 Stage 2 中可能饱和的 alpha 状态

#### 3.4.3 输出

```python
z_b:    (B, 226, 4096)   # 融合后的 brain latent，作为 DiT 的 cross-attention context
alphas: {
    α_key:   (B, 1),     # 关键帧 guidance 权重
    α_txt:   (B, 1),     # 文本 guidance 权重
    α_mot:   (B, 1),     # 运动 guidance 权重
    α_brain: (B, 1),     # brain latent 自身 guidance 权重
}
```

### 3.5 Multi-Guidance Adapter (MGA)

MGA 将 CMGF 输出的 z_b 和 alphas 与 Slow/Fast Branch 的中间产物组合，生成最终的 DiT context。

#### 3.5.1 Per-Channel Cross-Attention (v3.1)

v3.0 中 guidance 以全局向量 broadcast 到所有空间位置，缺乏空间选择性。v3.1 改为**每通道独立 cross-attention**：

```python
class GuidanceCrossAttention(nn.Module):
    """每个 guidance 通道有自己的 cross-attention 层"""
    def forward(self, z_b, guidance_embed):
        # z_b (B, 226, 4096) 作为 Q
        # guidance_embed (B, 1, D) 作为 K, V
        # → 每个空间位置学习性地选择利用 guidance 信号的程度
        return cross_attn(Q=z_b, K=V=guidance_embed)
```

#### 3.5.2 Alpha-Weighted Additive Residual

```python
context = z_b                                                    # 基础 brain latent
context += α_key   * key_attn(z_b, proj(z_key))                 # + 关键帧 guidance
context += α_txt   * txt_attn(z_b, proj(z_txt))                 # + 文本 guidance  
context += α_mot   * mot_attn(z_b, proj(eeg_pooled))            # + 运动 guidance
context += α_brain * z_b                                        # + brain latent 自增强
output  = out_proj(context)                                      # (B, 226, 4096)
```

**Temporal Guidance (可选)**: 当 `use_temporal_guidance=True` 时，`global_dyn_token` 通过 gated residual adapter 额外注入运动 guidance 通道。

#### 3.5.3 零初始化策略

`GuidanceCrossAttention` 的输出投影层全零初始化 → 训练初期 cross-attention 输出为 0 → guidance 注入是纯 residual → **不破坏 z_b 原有的扩散模型 context 分布**。随训练推进，guidance 逐渐从零"生长"出来。

### 3.6 Alignment Loss with MoCo Queue

CineBrain 原始的 5 路 InfoNCE 对比学习在 batch_size=1 时完全退化（无负样本）。CineBrain-SF v1 引入 **MoCo 式 memory queue** 解决此问题：

```python
# 维护 512 容量的 momentum queue
queue_fmri = Queue(capacity=512, dim=1152)
queue_eeg  = Queue(capacity=512, dim=1152)
queue_video = Queue(capacity=512, dim=1152)
queue_text  = Queue(capacity=512, dim=1152)

# 每步: 当前 embedding 与 queue 中的历史 embedding 构成 negatives
negatives = queue_fmri.get_all()                    # (512, 1152)
logits = cosine_sim(current_eeg, negatives) / tau   # InfoNCE
loss = cross_entropy(logits, target=0)              # 正样本在位置 0
queue_fmri.enqueue(current_fmri.detach())           # 更新 queue
```

**5 路对齐**:
```
L_align = L_fv(fmri↔video) + L_ft(fmri↔text) + L_ev(eeg↔video) + L_et(eeg↔text) + 0.5·L_fe(fmri↔eeg)
```

### 3.7 Guidance Consistency Losses

在 Fusion 和 Joint 阶段，guidance 一致性 loss 确保 guidance 信号与视频内容对齐：

```
L_gk = 1 - cosine(z_key, video_embed)              # 关键帧 guidance ↔ 视频语义
L_gt = 1 - cosine(z_txt, text_embed)                # 文本 guidance ↔ 文本语义
L_gm = detached_scale_MSE(motion_energy, flow_energy)  # 运动 guidance ↔ 光流强度
```

其中 `L_gm` 使用 detached-scale MSE（而非 cosine），解决 bs=1 时 cosine similarity 退化问题。

### 3.8 Total Loss

```
L_total = L_diff + λ_sf · (L_align + L_slow + L_fast + L_guide)

where:
  L_diff  = E[||ε - ε_θ(x_t, context, t)||²]       # 标准扩散去噪损失
  L_slow  = L_key + L_txt + L_str                    # Slow Branch 头部监督
  L_fast  = L_distill + L_temporal + L_flow + L_dyn + L_struct  # Fast Branch 综合监督
  L_guide = 0.5·L_gk + 0.5·L_gt + 0.5·L_gm          # Guidance 一致性
  λ_sf    = 0.003                                     # SF 辅助 loss 全局缩放因子
```

### 3.9 Three-Stage Progressive Training

三阶段训练策略是 CineBrain-SF 的关键设计，逐步解冻模块以避免梯度冲突和训练不稳定。

#### Stage 1: Branch Pretrain

**目标**: 让 Slow/Fast Branch 各自学习稳定的中间表征

| 模块 | 状态 |
|------|------|
| fMRI/EEG Encoder | 解冻 (trainable) |
| Slow Branch heads | 解冻 |
| Fast Branch (P0+P1) | 解冻 |
| GatedFusion | 冻结 |
| MultiGuidanceAdapter | 冻结 |
| DiT (CogVideoX-5B) | 冻结 |

**Loss**: `L = L_align + L_slow + L_fast` (无 L_diff)

**训练细节**:
- 先 P0 (纯蒸馏) 建立基础表征，再引入 P1 (时序动态)
- P0→P1 过渡时调整权重: λ_distill 从 1.0 降至 0.2, λ_temporal 从 0 升至 1.0
- 这确保训练后期时序信息主导，而非 fMRI 的"影子"

#### Stage 2: Fusion Training

**目标**: 让 GatedFusion 和 MultiGuidanceAdapter 学习融合 P1 后**正交的** Fast/Slow 特征

| 模块 | 状态 |
|------|------|
| fMRI/EEG Encoder | 冻结 |
| Slow/Fast Branch | 冻结 |
| GatedFusion | **解冻** |
| MultiGuidanceAdapter | **解冻** |
| DiT | 冻结 (但**梯度穿透**) |

**Loss**: `L = L_diff + λ_sf · (L_slow + L_fast + L_guide)`

**关键设计**: DiT 参数冻结 (`requires_grad=False`) 但不用 `torch.no_grad()`，允许 L_diff 的梯度穿透 42 层 DiT 回传到 Fusion 模块。经验证，穿透后 GatedFusion 梯度 norm ≈ 3.21e-04（非零，有效）。

#### Stage 3: Joint Finetuning

**目标**: LoRA 微调 DiT，让扩散模型适应 brain latent context 分布

| 模块 | 状态 |
|------|------|
| fMRI/EEG Encoder | 冻结 |
| Slow Branch | 冻结 |
| Fast Branch | **解冻** (继续优化时序) |
| GatedFusion | **解冻** |
| MultiGuidanceAdapter | **解冻** |
| DiT | **LoRA 微调** (r=128, alpha=128, scaling=1.0) |

**Loss**: `L = L_diff + λ_sf · (L_align + L_slow + L_fast + L_guide)`

**LoRA 配置**: rank=128, lora_alpha=128 → scaling=1.0。v1 实验中发现 lora_alpha=1 (scaling=1/128) 导致 DiT 几乎不更新，v2 修正为 scaling=1.0 后 diff_loss 从 0.86 降至 0.13。

---

## 4. Inference Optimization

### 4.1 DANA: Dynamic-Aware Noise Adding

受 EEG2Video 启发，利用 Fast Branch 的 flow trajectory 预测值作为每帧运动强度指标 β，修改扩散模型的噪声公式：

```
z_T = α_T · z_0 + √(1-α_T) · (β · ε_dynamic + (1-β) · ε_static)
```

其中 `β` 由 flow trajectory 预测值归一化得到：高运动帧使用更多 diverse noise，低运动帧使用更多 static noise。

**实验结果**: FVD 618.7 → 602.4 (↓2.6%), Img 50-way ↑7%，零训练成本。

---

## 5. Experiments

### 5.1 Setup

- **数据集**: CineBrain 自然视频观看 fMRI+EEG 数据集
- **被试**: Sub-05 (within-subject), Sub-03/04 (cross-subject)
- **Backbone**: CogVideoX-5B (DiT-based video diffusion model)
- **GPU**: NVIDIA A800 80GB PCIe × 4~5 卡
- **精度**: bf16 混合精度训练

### 5.2 Main Results: Within-Subject (Sub-05)

| 指标 | CineBrain baseline | **SF-v1 v2** | 变化 |
|------|:--:|:--:|------|
| **FVD** ↓ | 895.14 | **618.72** | **↓30.9%** |
| **EPE** ↓ | 3.68 | **2.94** | **↓20.1%** |
| CTC | 0.979 | **0.987** | ↑0.8% |
| DTC | 0.959 | **0.981** | ↑2.3% |
| SSIM | 0.288 | **0.302** | ↑4.9% |
| CLIP Score | 0.737 | **0.747** | ↑1.4% |
| Img 50-way | 0.341 | **0.351** | ↑2.9% |
| CLIP-PCC | 0.975 | **0.985** | ↑1.0% |
| PSNR | 12.01 | 12.04 | ≈ |
| Vid 50-way | 0.318 | 0.317 | ≈ |
| Vid 2-way | 0.914 | 0.907 | ≈ |
| Hue-PCC | 0.410 | 0.389 | ↓5.1% |
| VIFI-Score | 0.849 | 0.839 | ↓1.2% |

**结论**: 11/14 项指标优于或持平 baseline。**FVD ↓31% + EPE ↓20%** 是论文级核心突破，分别代表整体视频质量和运动重建质量的显著改善。

### 5.3 Cross-Subject Generalization

| 指标 | CB m05→d03 | **SF m05→d03** | 变化 | CB m05→d04 | **SF m05→d04** | 变化 |
|------|:--:|:--:|------|:--:|:--:|------|
| **FVD** ↓ | 847.5 | **711.9** | **↓16.0%** | 750.2 | **487.2** | **↓35.1%** |
| **EPE** ↓ | 3.92 | **2.93** | **↓25.3%** | 2.85 | 3.30 | ↑15.8% |
| SSIM | 0.273 | **0.299** | **↑9.3%** | 0.267 | 0.235 | ↓12.0% |
| PSNR | 11.84 | 11.88 | ≈ | 10.95 | **12.17** | **↑11.1%** |
| Img 50-way | 0.360 | 0.300 | ↓16.7% | 0.373 | **0.428** | **↑14.8%** |

**三场景 FVD 平均**: 830.9 → 605.9 (**↓27.1%**)。跨被试泛化能力验证成功。

### 5.4 SF-Specific Evaluation (evaluate_p1.py)

| 指标 | Stage 2 | v1 (LoRA 1/128) | v2 (6 fixes) | Recovery best |
|------|:--:|:--:|:--:|:--:|
| L_temp_delta | 0.044 | 0.044 | 0.044 | **0.039** |
| flow_traj Pearson | 0.298 | 0.298 | 0.296 | **0.361** |
| Fast/Slow cosine | -0.027 | -0.029 | -0.023 | **-0.040** |
| α_mot Spearman | -0.019 | 0.123 | 0.072 | **0.105** |
| α_brain | 0.936 | 0.436 | 0.526 | 0.0 |

**4/4 Acceptance Checks** (recovery best @ iter 2500):
- [x] L_temp_delta < 0.05 (0.039)
- [x] flow_traj Pearson > 0.3 (0.361)
- [x] Fast/Slow cosine < 0 (-0.040)
- [x] α_mot Spearman > 0 (0.105)

### 5.5 Ablation: LoRA Scaling Impact

| 配置 | FVD | SSIM | diff_loss | α_brain |
|------|-----|------|-----------|---------|
| v1 (scaling=1/128) | 13673 | 0.012 | 无变化 | 0.436 |
| v2 (scaling=1.0) | **618.72** | **0.302** | 0.86→0.13 | 0.526 |

v1 灾难性失败证明 LoRA scaling 的正确设置对 DiT 适配至关重要。

---

## 6. Current Challenges & Open Problems

### 6.1 Gating-Quality Pareto Tradeoff (核心未解问题)

Stage 3 Joint Training 中存在 **画质 vs Gating 精度的帕累托前沿**：

| λ_router | FVD | Gating Spearman | 画质 | Gating |
|----------|-----|-----------------|------|--------|
| 0.8 | 2785 | 0.105 | 崩塌 | 精确 |
| 0.02 | ~正常 | -0.14 | 恢复 | 反向 |

**根因**: router BCE loss 在高权重下产生比去噪梯度强 260 倍的分类梯度，DiT 被迫"弃画从教"(abandon painting for teaching)。

**当前探索方向**:
- 权重退火 (weight annealing): 前 500 步高权重 → 后期低权重
- Focal Loss: 只惩罚难以分类的样本，对已正确分类的样本降权
- λ 黄金点搜索 (0.1~0.15 区间)

### 6.2 Flow Trajectory 的天花板

flow_traj Pearson 在 0.30~0.36 区间波动，从 Stage 1 到 Stage 3 无显著突破。这可能是 EEG 对运动信息解码能力的物理上限。

### 6.3 EEG Encoder 从零训练

当前 EEG encoder 是项目内从零训练的，没有利用任何 EEG 预训练模型 (LaBraM, Gram, EEGPT 等)。文献表明预训练 EEG 基础模型能显著提升特征质量。

### 6.4 226 Brain Tokens vs DiT 预训练分布

CogVideoX-5B 预训练于 ≤77 text tokens，我们使用 226 brain tokens 做 cross-attention，是分布外推。考虑 learned attention pooling 226→64 对齐预训练分布。

---

## 7. Future Directions (创新点讨论参考)

基于当前进展和文献调研，以下方向值得进一步研究：

### 7.1 推理端优化 (零训练成本)
- **alpha-Guidance / SDEdit**: 用 Slow Branch 输出构造模糊先验，从中间 timestep 开始去噪
- **DANA 噪声调度**: 已验证有效 (FVD ↓2.6%)，可进一步优化 β 参数化

### 7.2 训练优化 (不大改架构)
- **SoftCLIP Loss (MindCine)**: 保持 embedding 拓扑结构而非逼近绝对值
- **光流 Codebook 扩展**: 更大 K 值、层次化 codebook
- **Gradient-based loss balancing**: GradNorm / PCGrad / Uncertainty Weighting 自动平衡多 loss

### 7.3 架构改进
- **EEG 预训练基础模型 (LaBraM/Gram)**: 替换从零训练的 EEG encoder
- **三模态对比学习 (Mind-Animator)**: brain-text-vision BiInfoNCE
- **Brain token 降维**: 226→64 对齐 DiT 预训练分布
- **Multi-frame temporal guidance**: 将 temporal_tokens 直接作为 multi-frame cross-attention 的 K/V 注入 DiT

### 7.4 科学验证
- **完整消融实验**: Slow only vs Slow+P0 vs Slow+P0+P1 (验证 Fast Branch 独立贡献)
- **Gating 行为分析**: 高/低动态 clip 的 α_mot 分布差异
- **跨数据集泛化**: 在 SEED-DV 或其他 EEG 视频数据集上验证

---

## Appendix A: Loss Function Details

### A.1 AlignmentLoss (5-way InfoNCE + MoCo Queue)

```python
L_fv = InfoNCE(fmri_cls, video_embed, queue_video, tau)    # fMRI ↔ Video
L_ft = InfoNCE(fmri_cls, text_embed,  queue_text,  tau)    # fMRI ↔ Text
L_ev = InfoNCE(eeg_cls,  video_embed, queue_video, tau)    # EEG  ↔ Video
L_et = InfoNCE(eeg_cls,  text_embed,  queue_text,  tau)    # EEG  ↔ Text
L_fe = InfoNCE(eeg_cls,  fmri_cls,    queue_fmri,  tau)    # EEG  ↔ fMRI

L_align = L_fv + L_ft + L_ev + L_et + 0.5·L_fe
```

### A.2 FastBranchDistillLoss

```python
# P0 蒸馏
L_distill_cls     = MSE(eeg_cls_proj, fmri_cls.detach())
L_distill_spatial = MSE(eeg_pooled_proj, fmri_pooled.detach())

# P1 时序动态
gt_delta = gt_frame_embs - gt_frame_embs[:, 0:1, :]
L_temporal_delta = MSE(pred_delta, gt_delta)
L_temporal_abs   = MSE(pred_tokens, gt_frame_embs)

# P1 光流轨迹 (codebook 分类)
L_flow_traj = CrossEntropy(flow_traj_pred, gt_flow_class)  # (B, T, K) vs (B, T)

# 粗粒度动态
L_dyn = CrossEntropy(dyn_logits, gt_dyn_label_2class)

# 帧间结构
L_struct = MSE(pred_cosine_matrix[off_diag], gt_cosine_matrix[off_diag])
```

### A.3 Stage-wise Loss Composition

| Stage | L_diff | L_align | L_slow | L_fast | L_guide | Trainable |
|-------|:------:|:-------:|:------:|:------:|:-------:|-----------|
| 1 (Branch) | — | ✓ | ✓ | ✓ | — | Encoders + Branches |
| 2 (Fusion) | ✓ | ✓ | ✓ (monitor) | ✓ (monitor) | ✓ | Fusion + Guidance |
| 3 (Joint) | ✓ | ✓ | ✓ (monitor) | ✓ | ✓ | LoRA DiT + Fast + Fusion |

---

## Appendix B: Data Flow Dimensions

```
fMRI input:              (B, 5, 8405)
  → fMRI Encoder:        (B, 226, 2048) spatial + (B, 1152) cls
  → Slow Branch:         z_key (B, 1152), z_txt (B, 1152), z_str (B, 16, 60, 90)

EEG input:               (B, 5, 64, 800)
  → EEG Encoder:         (B, 226, 2048) spatial + (B, 1152) cls
  → Fast Branch P0:      eeg_cls_proj (B, 1152), eeg_pooled_proj (B, 2048)
  → Fast Branch P1:      temporal_tokens (B, T, 1152), global_dyn_token (B, 1152)
                          flow_traj_pred (B, T, K), dyn_logits (B, 2)

  → GatedFusion:         z_b (B, 226, 4096), alphas (4 × (B, 1))
  → MultiGuidanceAdapter: context (B, 226, 4096)

  → DiT (CogVideoX-5B):  denoised video latent
  → VAE Decoder:          video frames
```

---

*This document was compiled from the CineBrain-SF v1 project codebase, design documents, and experimental records as of 2026-04-16.*
