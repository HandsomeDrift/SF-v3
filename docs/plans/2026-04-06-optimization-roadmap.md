# SF-v1 优化路线图 — 复盘与调研

**日期**: 2026-04-06
**背景**: Stage 3 v2 评估完成（FVD ↓31%, EPE ↓20% vs CineBrain baseline），跨被试推理进行中。趁此窗口对当前系统做全面复盘和文献调研，规划后续优化方向。

---

## 一、当前系统总结

### 已验证的核心成果

| 指标 | CineBrain baseline | SF-v1 v2 (Sub-05) | 变化 |
|------|:--:|:--:|------|
| FVD ↓ | 895.14 | 618.72 | **↓30.9%** |
| EPE ↓ | 3.68 | 2.94 | **↓20.1%** |
| CTC | 0.979 | 0.987 | ↑ |
| DTC | 0.959 | 0.981 | ↑ |
| SSIM | 0.288 | 0.302 | ↑ |
| CLIP Score | 0.737 | 0.747 | ↑ |
| Vid 50-way | 0.318 | 0.317 | ≈ |
| Hue-PCC | 0.410 | 0.389 | ↓5% |

11/14 项指标优于或持平 baseline。Slow-Fast 假设验证成功。

### 三阶段训练体系

| Stage | 可训练模块 | 核心 Loss | 状态 |
|-------|-----------|----------|------|
| 1 (Branch) | EEG Encoder + Fast Branch + fMRI Encoder + Slow Branch | L_slow + L_fast (P0蒸馏 + P1时序) | 完成 |
| 2 (Fusion) | GatedFusion + MultiGuidanceAdapter | L_diff + L_guide | 完成 |
| 3 (Joint) | LoRA DiT + Fast Branch + Fusion | L_diff + λ_sf × sf_total | 完成 |

---

## 二、内部代码审查：已知问题与技术债

### 2.1 架构层面

#### [A-1] GatedFusion 信息瓶颈（高优先）
- **位置**: `gated_fusion.py:71-85`
- **问题**: slow+fast concat→共享线性投影→只取前 S 个 tokens（slow 侧）作为输出 z_b
- **影响**: Fast Branch 信息只能通过 transformer 内部的 cross-attention 间接影响输出，存在严重的信息瓶颈
- **证据**: alpha_mot Spearman 在 Stage 2 塌缩到 -0.019，说明运动信号在融合中丢失
- **HANDOFF 标记**: M-01（已知技术债）

#### [A-2] Guidance 全局广播无空间选择性（高优先）
- **位置**: `multi_guidance.py:72-95`
- **问题**: z_key、z_txt、eeg_pooled 都是 (B, D) 全局向量，unsqueeze(1) 后 broadcast 到所有 226 个空间位置
- **影响**: 关键帧 guidance 应该聚焦早期帧，运动 guidance 应该聚焦高运动区域，但当前所有位置收到相同信号
- **结果**: alpha_brain 主导（≈0.53），DiT 主要靠 z_b 本身而非精细 guidance

#### [A-3] L_align 在 batch_size=1 下完全失效（高优先）
- **位置**: `sf_losses.py:9-22`（AlignmentLoss）
- **问题**: InfoNCE 对比 loss 需要 in-batch negatives，bs=1 时无负样本
- **影响**: 5 路对齐 loss（L_fv, L_ev, L_ft, L_et, L_fe）全为 0，等于跨模态对比学习从未生效
- **状态**: P1 设计文档 §11 已识别，计划用 MoCo 式 memory queue 解决，但未实现

#### [A-4] 226 Brain Tokens vs DiT 预训练分布
- **问题**: CogVideoX-5B 预训练于 ≤77 text tokens，我们用 226 brain tokens 做 cross-attention，是分布外推
- **影响**: attention pattern 可能未被充分学习，DiT 处理长 context 效率低
- **可选方案**: learned attention pooling 226→64

#### [A-5] L_gm（运动 guidance loss）未实现
- **位置**: `sf_losses.py:199-219`（GuidanceLoss）
- **问题**: config 中定义了 L_gm，但 forward() 只计算 L_gk + L_gt，运动 guidance 无直接监督
- **影响**: alpha_mot 只通过间接的 L_diff 反传获得梯度，学习信号弱

#### [A-6] Causal Mask 可能过度约束
- **位置**: `temporal_dynamics.py:101-104`
- **问题**: 严格 causal mask 限制帧 t 只能 attend 到 ≤t 的帧，但 temporal queries 是抽象表示，因果约束可能不必要
- **可选方案**: 移除 causal mask，或叠加 random sparse mask 防 shortcut

### 2.2 训练层面

#### [T-1] Fast Branch 光流回归先天困难
- **证据**: flow_traj Pearson 仅 0.298（Stage 1 到 Stage 3 无改善）
- **根因**: EEG 无法预测像素级运动，RAFT 光流统计量与神经信号之间存在语义鸿沟
- **这是 P0 阶段就发现的问题**，当时绕过了（从分类改为蒸馏），但运动回归这条路始终没走通

#### [T-2] Stage 1/2 基于早期不完整理解
- Stage 1 训练时还没发现 LoRA scaling、动态缩放等问题
- Stage 2 训练时 L_align 失效（bs=1）、L_gm 未实现，fusion 学到的权重可能不是最优的
- 理论上应该在 fix 之后重新训练，但时间成本高

#### [T-3] EEG Encoder 从零训练
- 当前 EEG encoder 是项目内从零训练的，没有利用任何 EEG 预训练模型
- 文献表明预训练 EEG 基础模型（LaBraM, Gram, EEGPT 等）能显著提升特征质量

#### [T-4] P1 temporal_guidance 闲置
- `multi_guidance.py` 中 `use_temporal_guidance` 选项已实现但 config 中 disabled
- P1 的 `global_dyn_token` 可以作为额外 guidance 信号注入，zero cost 启用

---

## 三、文献调研：竞品核心技术

### 3.1 论文摘要

#### NeuroClips (NeurIPS 2024 Oral)
- **模态**: fMRI → Video | **Backbone**: AnimateDiff v3 + SD 1.5
- **核心**: Inception Extension（单 TR → 多帧 embedding）+ **三重 Guidance**
  - alpha-Guidance: 用模糊视频作为扩散中间态，只需 0.3T 步去噪（SDEdit 思路）
  - beta-Guidance: ControlNet (SparseCtrl) 关键帧注入
  - gamma-Guidance: BLIP-2 caption 作为文本提示
- **结果**: cc2017 上 SSIM 0.390 (+128% vs MinD-Video)
- **启示**: **alpha-Guidance 是推理端零成本提升的最佳选择**

#### EEG2Video (NeurIPS 2024)
- **模态**: EEG → Video | **Backbone**: Inflated SD
- **核心**: DANA (Dynamic-Aware Noise Adding) — 修改噪声公式 z_T = α_T·z0 + √(1-α_T)·(β·ε_s + (1-β)·ε_d)，β 由光流得分动态控制
- **结果**: SEED-DV 上 2-way 79.8%
- **启示**: **我们的 flow_trajectory 可以直接作为 DANA 的 β 参数**

#### DecoFuse (arXiv 2025)
- **模态**: fMRI → Video | **Backbone**: DragNUWA
- **核心**: What/Where/How 三重分解 + **光流 Codebook**（K-means 离散化光流空间，将运动预测从回归转为分类）
- **启示**: **Codebook 方案可能根治我们 Fast Branch 光流回归困难的问题**

#### MindCine (arXiv 2025)
- **模态**: EEG → Video | **Backbone**: T2V model
- **核心**: **EEG 预训练基础模型** (LaBraM/Gram) + CausalSeq + **SoftCLIP Loss**（软对比对齐，比较 embedding 间相似度比值而非绝对距离）
- **结果**: SEED-DV 上 Video 2-way 0.818
- **启示**: 预训练 EEG encoder + SoftCLIP 两项技术可直接借鉴

#### DynaMind (arXiv 2025)
- **模态**: EEG → Video | **Backbone**: SD v1.4 + Tune-A-Video
- **核心**: TDA temporal blueprint + L_Struct + **Temporal Blueprint 初始化**（x_T = ε + α·U(z_B)，不从纯噪声开始）
- **结果**: SEED-DV 上 Video 40-way 0.284
- **启示**: L_Struct 已借鉴；Blueprint 初始化与 alpha-Guidance 思路一致

#### Mind-Animator (ICLR 2025)
- **模态**: fMRI → Video | **Backbone**: Inflated SD
- **核心**: CMG (Consistency Motion Generator) 使用 **Sparse Causal Self-Attention**（causal + random sparse mask 防 shortcut）+ 三模态对比学习（brain-text-vision BiInfoNCE）
- **结果**: cc2017 上 SSIM 0.321, EPE 5.422
- **启示**: Sparse mask 极易实现；三模态对比可恢复我们失效的 L_align

### 3.2 我们已经走在前面的方面

| 技术 | 我们的实现 | 相关论文 |
|------|-----------|---------|
| Slow-Fast 分离 | Slow Branch (fMRI) + Fast Branch (EEG) | DynaMind, DecoFuse 独立验证 |
| CausalSeq | TemporalDynamicsDecoder + causal mask | MindCine 也采用 |
| L_Struct | sf_losses.py 中 cosine similarity matrix | DynaMind 独立提出 |
| fMRI+EEG 多模态 | 双分支融合 | 多数论文只用单模态 |
| CogVideoX-5B DiT | 最强视频生成 backbone | 竞品用 SD 1.5 / AnimateDiff |

---

## 四、优化路线图

### Phase 0: 推理端优化（不需重训，1-2 周）

**性价比最高的方向。用已有模型，改推理策略即可验证。**

#### P0-1. alpha-Guidance / SDEdit 式推理
- **思路**: 用 Slow Branch 输出构造模糊先验 x_init，从中间 timestep（~0.3T ≈ 15 步）开始去噪，而非从纯噪声 51 步
- **来源**: NeuroClips + DynaMind
- **预期收益**: 减少去噪步数（51→~35）；提升低频语义一致性（Vid 50-way, Hue-PCC）
- **实现**: 修改 `sample_brain_va.py` 的 sampling pipeline
- **难度**: 中（需理解 VPSDEDPMPP2MSampler 的噪声调度）
- **验证**: 在已有 540 样本上直接对比

#### P0-2. DANA 动态噪声调度
- **思路**: 用 Fast Branch 的 `flow_trajectory` 预测值作为每帧运动强度指标 β。β 高（运动强）→ 更多 diverse noise；β 低（静态）→ 更多 static noise
- **来源**: EEG2Video
- **公式**: z_T = α_T·z0 + √(1-α_T)·(β·ε_d + (1-β)·ε_s)
- **预期收益**: EPE 进一步改善，运动丰富场景重建质量提升
- **实现**: 修改 diffusion sampler 的 noise 生成逻辑
- **难度**: 中
- **验证**: mini50 快速验证 → 全量对比

### Phase 1: 快速训练优化（不大改架构，2-3 周）

#### P1-1. Sparse Causal Attention
- **思路**: 在 TemporalDynamicsDecoder 的 causal mask 上叠加 random sparse mask（每层随机 drop 30% attend 位置）
- **来源**: Mind-Animator
- **预期收益**: temporal decoder 泛化性改善，防止 shortcut
- **实现**: `temporal_dynamics.py` 加几行 mask 逻辑
- **难度**: **极低**（1-2 小时）
- **验证**: mini500 对比有无 sparse mask

#### P1-2. 启用 temporal_guidance
- **思路**: 将 config 中 `use_temporal_guidance` 设为 true，把 P1 的 `global_dyn_token` 作为额外 guidance 信号注入
- **实现**: 改 yaml 配置
- **难度**: **极低**
- **验证**: Stage 3 重训对比

#### P1-3. 光流 Codebook（分类替代回归）
- **思路**: 对训练集所有光流向量做 K-means 聚类（K=64~256），Fast Branch 的 flow_trajectory 从 MSE 回归改为 CrossEntropy 分类
- **来源**: DecoFuse
- **动机**: 根治 EEG → 像素级运动回归的语义鸿沟（flow_traj Pearson 卡在 0.298）
- **实现**:
  1. 提取训练集光流，K-means 聚类生成 codebook
  2. 修改 `fast_branch.py` 的 flow trajectory head（回归→分类）
  3. 修改 `sf_losses.py` 中 L_flow_traj（MSE→CE + 概率加权）
  4. 从 Stage 1 重训 Fast Branch
- **难度**: 中
- **验证**: flow_traj accuracy (分类) vs flow_traj Pearson (回归)

#### P1-4. SoftCLIP Loss + Queue-based 对比学习
- **SoftCLIP**: 比较 brain embedding 间的相似度比值与 CLIP embedding 间的比值（保持拓扑结构而非逼近绝对值）
- **Queue**: 实现 MoCo 式 memory queue（维护 256~1024 个历史 negative embeddings），解决 bs=1 下 L_align 失效
- **来源**: MindCine + MoCo
- **预期收益**: 恢复 5 路 L_align 信号，改善 brain-visual 空间对齐
- **实现**: 修改 `sf_losses.py` 的 AlignmentLoss，加入 queue 和 SoftCLIP 计算
- **难度**: 中
- **验证**: Stage 1 重训，观察 L_align 是否有非零值

### Phase 2: 架构改进（较大改动，3-4 周）

#### P2-1. GatedFusion 重设计（最高优先）
- **当前**: concat→shared projection→只取 slow 侧 tokens
- **新方案**: 分离投影 + cross-attention mixing
  ```
  slow_feat → proj_slow → Q
  fast_feat → proj_fast → K, V
  z_fused = CrossAttention(Q, K, V) + slow_feat (residual)
  ```
- **预期**: Fast Branch 信息充分传递，alpha_mot 分化改善
- **实现**: 重写 `gated_fusion.py`
- **难度**: 中-高
- **注意**: 需要从 Stage 2 重训

#### P2-2. Guidance 注入改为 cross-attention
- **当前**: 全局向量 broadcast 到所有 226 位置
- **新方案**: spatial tokens 做 query, guidance embeddings 做 key/value 的 cross-attention
  ```
  context = CrossAttention(Q=spatial_tokens, K=V=[z_key, z_txt, z_mot]) + z_b
  ```
- **预期**: 不同空间位置选择性利用不同 guidance，改善空间精度
- **难度**: 中

#### P2-3. 实现 L_gm（运动 guidance loss）
- **思路**: 比较 motion guidance 输出与光流 codebook 特征（如果 P1-3 已实现）或光流统计量的一致性
- **预期**: alpha_mot 有直接梯度信号
- **难度**: 低-中

#### P2-4. Brain tokens 降维 226→64
- **思路**: learned attention pooling 或 top-K ROI selection
- **预期**: 对齐 DiT 预训练分布，减少 attention 复杂度
- **难度**: 中

### Phase 3: 基础设施升级（长期，4-6 周）

#### P3-1. EEG 预训练基础模型
- **思路**: 替换从零训练的 EEG encoder，用 LaBraM / Gram / EEGPT 作为 backbone
- **来源**: MindCine
- **预期**: EEG 特征质量天花板提升，Fast Branch 所有指标改善
- **难度**: 中（需适配 CineBrain EEG 数据格式：5 trials × 64 channels × 800 samples）
- **前置**: 评估预训练模型在 CineBrain 数据上的 feature quality

#### P3-2. 三模态对比学习（brain-text-vision）
- **思路**: fMRI/EEG embedding + text caption embedding + video frame embedding 三路 BiInfoNCE
- **来源**: Mind-Animator
- **预期**: 增强语义空间质量，改善 Vid 50-way 和 CLIP Score
- **难度**: 中-高

#### P3-3. 全流程重训
- 基于 Phase 1-2 的所有改进，从 Stage 1 重新开始完整三阶段训练
- 重点: 光流 codebook + SoftCLIP + 新 GatedFusion + Guidance cross-attention

---

## 五、推荐执行顺序

```
                                    时间线
Week 1-2    ┌─ P0-1 alpha-Guidance ─────────┐
            ├─ P0-2 DANA 噪声 ──────────────┤  推理端验证
            ├─ P1-1 Sparse Causal (1h) ─────┤  （不需重训）
            └─ P1-2 temporal_guidance (cfg) ─┘

Week 3-4    ┌─ P1-3 光流 Codebook ──────────┐
            └─ P1-4 SoftCLIP + Queue ───────┘  快速训练验证
                                                （Stage 1 局部重训）

Week 5-7    ┌─ P2-1 GatedFusion 重设计 ────┐
            ├─ P2-2 Guidance cross-attn ────┤  架构改进
            ├─ P2-3 L_gm 实现 ─────────────┤  （Stage 2/3 重训）
            └─ P2-4 Token 降维 ─────────────┘

Week 8-12   ┌─ P3-1 EEG 预训练模型 ────────┐
            ├─ P3-2 三模态对比 ─────────────┤  基础设施
            └─ P3-3 全流程重训 ─────────────┘
```

**关键依赖关系**:
- P0-2 (DANA) 依赖 Fast Branch 的 flow_trajectory 输出（已有）
- P1-3 (光流 Codebook) 需要先提取训练集光流并聚类
- P2-1 (GatedFusion) 需要从 Stage 2 重训，之后的 Stage 3 也要重跑
- P3-3 (全重训) 应等 Phase 1-2 的改进全部验证后再执行

---

## 六、关键决策点

1. **alpha-Guidance vs DANA**: 两者都是推理端优化，可以独立验证也可以组合。alpha-Guidance 改变初始状态（从中间开始），DANA 改变噪声组成（static/diverse 比例）。建议先验证 alpha-Guidance（改动更小），再叠加 DANA。

2. **光流 Codebook 的 K 值**: K 太小（16）会丢失运动细节，K 太大（512）回到回归问题。建议从 K=64 开始，消融 32/128/256。

3. **是否从 Stage 1 全重训**: Phase 2 的架构改动（特别是 P2-1 GatedFusion）需要从 Stage 2 重训。但如果 P1-3 (Codebook) 和 P1-4 (SoftCLIP) 效果显著，Stage 1 也值得重做。建议 Phase 1 验证后再决定。

4. **EEG 预训练模型的兼容性**: CineBrain 的 EEG 格式（5 trials × 64 channels × 800 samples @ 200Hz）需要检查与 LaBraM/Gram 的兼容性（采样率、通道数、时长）。

---

## 七、参考文献

| 论文 | 会议 | 核心贡献 | 借鉴技术 |
|------|------|---------|---------|
| NeuroClips | NeurIPS 2024 Oral | Inception Extension + 三重 Guidance | alpha-Guidance (P0-1) |
| EEG2Video | NeurIPS 2024 | Seq2Seq + DANA 噪声调度 | DANA (P0-2) |
| DecoFuse | arXiv 2025.04 | What/Where/How + 光流 Codebook | Codebook (P1-3) |
| MindCine | arXiv 2025.01 | CausalSeq + SoftCLIP + EEG 预训练 | SoftCLIP (P1-4), EEG FT (P3-1) |
| DynaMind | arXiv 2025.09 | TDA Blueprint + L_Struct | Blueprint 初始化 (P0-1) |
| Mind-Animator | ICLR 2025 | Sparse Causal + 三模态对比 | Sparse mask (P1-1), 三模态 (P3-2) |
| NeuroCine | arXiv 2024.02 | Dependent Prior Noise | 帧间噪声关联 |
| SemVideo | arXiv 2026.02 | 层次化语义 Mining | 长期参考 |
| MindEye2 | ICML 2024 | fMRI→CLIP 蒸馏 + diffusion prior | 基础参考 |
