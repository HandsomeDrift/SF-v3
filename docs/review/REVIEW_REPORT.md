# CineBrain-SF v1 深度 Code Review 报告

**日期**: 2026-04-04
**范围**: 全项目代码、配置、训练日志、评估结果、设计文档
**方法**: 6 个并行审查 agent（数据流/模型架构/Loss训练/配置评估/日志分析/设计对比）+ 交叉验证

---

## 1. 项目整体理解

### 项目在做什么

CineBrain-SF v1 是一个 fMRI+EEG 多模态脑信号到视频重建系统。核心假设：显式的 Slow-Fast 角色分配（fMRI=语义/结构，EEG=运动/动态）优于统一的多模态融合。基于 CogVideoX-5b（Diffusion Transformer）最小侵入式扩展。

### 核心训练链路

```
fMRI (5, 8405) ─→ fMRI Encoder ─→ SlowBranch ──┐
                                                 ├→ GatedFusion ─→ MultiGuidance ─→ context (B, 226, 4096) ─→ DiT (CogVideoX-5b) ─→ L_diff
EEG (5, 64, 800) ─→ EEG Encoder ─→ FastBranch ─┘     ↓
                                     ├→ P0 蒸馏 ─→ L_distill         GatedFusion ─→ L_guide
                                     └→ P1 时序 ─→ L_temporal
```

训练分 3 阶段：
- **Stage 1**: 训练 Fast Branch（EEG encoder + P0蒸馏 + P1时序），DiT frozen
- **Stage 2**: 训练 GatedFusion + MultiGuidanceAdapter，branches + DiT frozen
- **Stage 3**: Joint — LoRA 微调 DiT + 解冻 Fast Branch + Fusion

### 核心推理链路

`fMRI + EEG → Encoders → SlowBranch + FastBranch → GatedFusion → MultiGuidance → context → DiT (DDIM 50 steps + DynamicCFG) → VAE Decode → Video`

### 核心评估链路

`生成视频 vs GT 视频 → SSIM, PSNR, CLIP-Score, FVD, CTC`（标准视频质量指标，无 brain-specific 指标）

---

## 2. 最可能影响性能的关键问题（Top Issues）

### Issue #1: LoRA alpha 未设置，scaling = 1/128，DiT 几乎不更新

- **文件**: `configs/sf_v1/cinebrain_sf_v1_model.yaml:58`, `sat/model/finetune/lora2.py:185,199`
- **问题**: 配置只设了 `r: 128`，未设 `lora_alpha`。`LoraMixin` 默认 `lora_alpha=1`，导致 `scaling = lora_alpha / r = 1/128 = 0.0078`。LoRA 对 DiT 的输出贡献被缩小 128 倍。
- **为什么导致效果差**: Stage 3 的核心目标是让 DiT 学会利用 brain conditioning 生成视频。scaling=0.0078 意味着即使 LoRA 参数有较大梯度更新，对 DiT 的实际输出影响也微乎其微。**这是 Stage 3 所有 loss 960 iter 完全平坦的最可能单一原因。**
- **概率**: 极高（代码确认）
- **影响程度**: **极高** — 直接导致 Stage 3 训练无效
- **修改建议**: 添加 `lora_alpha: 128`（使 scaling=1.0），这是 CogVideoX 微调的标准配置

### Issue #2: 动态缩放使所有 SF Loss lambda 权重完全失效

- **文件**: `sgm/modules/diffusionmodules/loss.py:320-323`
- **问题**: `scale = diff_loss.detach() / (sf_total.detach() + 1e-8); sf_total = sf_total * scale`。将 sf_total 强行缩放到与 diff_loss 相同量级。
- **为什么导致效果差**: 设计文档中精心调节的 lambda 权重调度（Stage A: λ_distill=1.0, Stage C: λ_distill=0.2 等）**完全无效**。sf_total 的实际大小只由 diff_loss 决定。更严重的是，冻结模块的 loss 值（Stage 2 中的 L_fast）也参与 sf_total，稀释了可训练模块 loss 的有效梯度。
- **概率**: 确认（代码和日志双重验证）
- **影响程度**: **极高** — 破坏了整个多 loss 权重体系
- **修改建议**: 删除动态缩放，使用固定 `lambda_sf` 权重：`return diff_loss + lambda_sf * sf_total`

### Issue #3: fMRI 和 EEG 原始数据完全没有归一化

- **文件**: `data_video.py:611,622`
- **问题**: fMRI `(5, 8405)` 和 EEG `(5, 64, 800)` 从 .npy 直接加载，无任何 z-score 或 min-max 归一化。encoder 内部也没有输入归一化层。
- **为什么导致效果差**: fMRI 不同 run 之间的 baseline drift、scanner noise 导致幅度差异很大。EEG 更甚（电极阻抗变化、运动伪迹）。文献中几乎所有 fMRI/EEG decoding 工作都做 per-voxel/per-channel z-score。**不归一化导致 encoder 输入分布不稳定，学习效率严重下降。**
- **概率**: 高（需验证 .npy 文件本身是否已预处理）
- **影响程度**: **高** — 影响所有阶段的训练效果
- **修改建议**: 在 `get_item_func()` 中添加 per-sample z-score，或使用全局统计量归一化。需先检查 .npy 文件中的数据分布确认。

### Issue #4: Guidance 注入方式 — 全局向量广播 + alpha_brain 主导

- **文件**: `sgm/modules/encoders/multi_guidance.py:70-96`
- **问题**: guidance 向量 (z_key, z_txt, eeg_pooled_proj) 都是 (B, D) 的全局向量，通过 `.unsqueeze(1)` 广播加到所有 226 个 spatial token。加上 `alpha_brain ≈ 0.94`（Stage 2 观察值），实际 context ≈ `1.94 * z_b + 小扰动`。
- **为什么导致效果差**: 所有 guidance head 的输出对 DiT 的影响极小，模型几乎只看到 z_b 本身。空间维度上没有任何位置特异性。精心设计的 keyframe/text/motion guidance 被 brain_latent 主导的残差连接淹没。
- **概率**: 高（Stage 2 评估数据确认 alpha_brain=0.94）
- **影响程度**: **高** — guidance 信号被稀释，DiT 无法有效利用
- **修改建议**: (1) 用 cross-attention 注入 guidance 而非加法广播；(2) 去掉 `alpha_brain * z_b` 的残差连接或改为乘性 gate

### Issue #5: 226 brain tokens 远超 DiT 预训练的 text 长度

- **文件**: `dit_video_concat_fmri.py:768-769,456-517`
- **问题**: CogVideoX-5b 预训练时 text tokens 通常 64-77 个。现在 context 有 226 个 brain tokens，数量差 3-4 倍。RoPE 位置编码超出预训练范围，AdaLN 的 text gate 需要适应完全不同的序列长度。
- **为什么导致效果差**: DiT 的 self-attention 和 cross-attention 需要大幅适应新的 sequence length。LoRA 的适应能力有限（特别是 scaling=1/128 时几乎为零）。226^2 ≈ 51K attention pairs vs 原始 ~4K，计算特性完全不同。
- **概率**: 中高（架构决策，非 bug）
- **影响程度**: **中高** — 增加了 DiT 适配难度
- **修改建议**: 考虑对 brain tokens 做 learned pooling 降到 32-64 个，或使用 cross-attention adapter

---

## 3. 严重实现问题（P0）

### P0-1: LoRA scaling = 1/128
*(详见 Issue #1)*

### P0-2: 动态缩放使 lambda 失效
*(详见 Issue #2)*

### P0-3: fMRI/EEG 无归一化
*(详见 Issue #3，需先验证 .npy 文件数据分布)*

### P0-4: `gt_dyn_label_2class` 键名不匹配 — L_dyn 永远不计算

- **文件**: `data_video.py:674` vs `sf_losses.py:188`
- **问题**: 数据管道映射 `dyn_class_3 → gt_dynamics_class`，但 loss 查找 `gt_dyn_label_2class`。条件永远为 False。
- **额外**: 即使名字对上，数据是 3-class 但 head 输出 2-class，维度也不匹配。
- **影响**: `coarse_dyn_head` 完全无监督，`lambda_dyn=0.1` 权重浪费。
- **修改**: 在 `data_video.py` 的 `_name_map` 中修复映射并做 3→2 类转换。

### P0-5: Text caption 随机截断与固定 gt_text_embed 不匹配

- **文件**: `data_video.py:626-629`
- **问题**: 每次加载同一样本，caption 随机截取 2 句。但 `gt_text_embed` 是离线提取的固定 embedding（可能用完整 caption）。模型试图用随机截断 text 拟合固定 target，引入系统性 label noise。
- **影响**: L_txt 监督信号含大量噪声，降低 Slow Branch 文本对齐能力。
- **修改**: 训练时使用完整 caption，或离线 target 提取时使用同样的截断策略。

### P0-6: `_clip_loss` 缺少 L2 归一化

- **文件**: `sf_embedder.py:269-273`
- **问题**: `_clip_loss` 计算 `logit_scale * feat_a @ feat_b.T` 但未对 features 做 L2 归一化。feature norm 可能达到 30-50，`logit_scale ≈ 14.3`，logits 值可能达到 ~35000，softmax 完全退化。对比之下 `AlignmentLoss.contrastive()` 正确做了归一化。
- **注意**: 当前 `mode: infer` 导致 `_compute_clip_loss` 不被调用，所以**此 bug 目前不触发**。但如果启用 clip loss mode，会导致灾难性的梯度行为。
- **修改**: 添加 `feat_a = F.normalize(feat_a, dim=-1)`。

---

## 4. 中优先级问题（P1）

### P1-1: Stage 3 所有 loss 960 iter 完全平坦

- **证据**: log 分析显示 total loss 从 0.256→0.268 (+4.7%)，L_slow/L_guide/所有 temporal loss 均无下降趋势。
- **根因**: Issue #1 (LoRA scaling) + Issue #2 (动态缩放) + gradient_clipping=0.1 的叠加效应。
- **影响**: 当前正在运行的训练（预计 04-05 01:40 完成）可能完全浪费 GPU 时间。

### P1-2: Stage 2 L_guide 几乎不下降 (<1%)

- **证据**: L_guide 从 0.162→0.161 (2000 iter)。但 alpha 分化确实出现了 (brain=0.94, txt=0.83)。
- **根因**: GatedFusion + MultiGuidanceAdapter 可训练参数极少（~90个），在 gradient_clipping=0.1 下有效梯度微弱。动态缩放进一步压缩了 L_guide 的权重。verify_grad.log 显示 GatedFusion gradient total norm 仅 3.21e-04。

### P1-3: gradient_clipping=0.1 过于激进

- **文件**: 各训练 yaml `deepspeed.gradient_clipping: 0.1`
- **问题**: 对 1.1B trainable params 的模型，0.1 阈值可能将所有有效梯度截断。与 LoRA scaling 问题叠加，DiT 的实际更新可能接近零。
- **修改**: 提高到 0.5-1.0。

### P1-4: 无 LR warmup / scheduler

- **文件**: 所有训练配置
- **问题**: 所有阶段均使用固定 lr=1e-4，无 warmup/cosine decay。LoRA 从零初始化，首 step 就用全 LR 可能破坏 Fusion 稳定状态。
- **修改**: 添加 200 steps 线性 warmup + cosine decay。

### P1-5: AlignmentLoss 在 bs=1 下恒为 0

- **文件**: `sf_losses.py:16-22`
- **问题**: batch_size=1 时 InfoNCE 退化为 -log(1) = 0。5 个 alignment loss 全为 0，编码器无跨模态对齐信号。
- **修改**: 短期设 lambda=0 禁用；长期实现 queue-based negatives。

### P1-6: L_gm (motion guidance loss) 完全缺失

- **文件**: `sf_losses.py:197-219`
- **问题**: `GuidanceLoss` 接收 `lambda_gm` 但 `forward()` 只计算 L_gk 和 L_gt，L_gm 从未计算。Motion guidance 无直接监督。
- **修改**: 定义蒸馏特征与运动 target 的 consistency loss，或移除 `lambda_gm` 参数避免混淆。

### P1-7: Train-Valid gap 达 300-400 倍

- **证据**: Stage 2 train total=0.25 vs valid total=80-160（300x+）
- **根因**: (1) eval_iters 太小（5 个样本），方差极大；(2) 可能的 BatchNorm bf16 兼容问题；(3) 训练集与验证集分布差异。
- **修改**: 增大 eval_iters 到至少 50。

### P1-8: P1 训练中 P0 蒸馏知识被严重遗忘

- **证据**: P1 validation `L_distill_spatial` 从 215 暴涨到 655 再回落到 283。
- **根因**: P1 temporal loss 与 P0 distill loss 竞争——temporal decoder 将 EEG 特征拉离 fMRI 空间（这正是设计目标），但代价是蒸馏知识损失。
- **影响**: 如果 Stage 2/3 的 brain guidance 依赖 distill 特征，质量会下降。

### P1-9: GatedFusion 先 concat 再投影 + 只取 slow-aligned tokens

- **文件**: `gated_fusion.py:71-72, 84-85`
- **问题**: slow/fast 先 concat 为 4096-dim，投影到 2048-dim，transformer 处理后**只取前 S 个 token（slow-aligned）**。Fast branch 信息只能通过投影层间接影响输出，被 slow 主导。
- **修改**: 分别投影后让 transformer 做跨模态交互。

### P1-10: Delta loss 第一帧恒为 0

- **文件**: `sf_losses.py:155-157`
- **问题**: `delta_t = z_t - z_1` 在 t=1 时恒为 0，MSE loss 贡献为 0。T=9 帧中 1/9 的 loss 无意义。
- **修改**: 从 t=2 开始计算 delta loss。

### P1-11: Causal mask 使早期 temporal query 孤立

- **文件**: `temporal_dynamics.py:101-104`
- **问题**: causal mask 下 t=1 只能 attend to 自己，完全失去 inter-frame 信息传递。
- **修改**: 考虑只在 cross-attention 中保持因果约束，self-attention 使用 full attention。

### P1-12: Frozen 模块 loss 仍参与 sf_total 并影响动态缩放

- **文件**: `loss.py:266-268`
- **问题**: Stage 2 中 Fast Branch 冻结，但 L_fast 仍计算并加入 sf_total。通过动态缩放机制，这些不可优化的 loss 值稀释了可训练 loss (L_guide) 的有效梯度。
- **修改**: 各阶段只累加可训练模块的 loss。

### P1-13: Gating alpha_mot 动态分化从 P1 的 PASS 退化到 Stage 2 的 FAIL

- **证据**: P1 Spearman=0.108 (p=0.012) → Stage 2 Spearman=-0.019 (p=0.66)。
- **根因**: Stage 2 冻结 Fast Branch + GatedFusion 学会抑制 alpha_mot (0.47 vs P1 的 0.52)。
- **影响**: Fast Branch 的运动引导在 Fusion 后形同虚设。

---

## 5. 次要问题（P2）

| # | 文件 | 问题 | 说明 |
|---|------|------|------|
| P2-1 | `eeg_encoder_custom.py` | TCN 在 pool 后运行 (800→226)，高频 EEG 信息已丢失 | 应在 pool 前做 TCN |
| P2-2 | `eeg/fmri_encoder_custom.py` | Pre-norm Transformer 缺 final LayerNorm | 标准做法应在 stack 后加 LN |
| P2-3 | `temporal_conv.py` | TemporalAttentionPool 用 nn.MHA，bf16 下可能有 bug | temporal_dynamics.py 专门实现了 BF16SafeAttention |
| P2-4 | `slow_branch.py` | StructureHead 354M 参数 (Linear 4096→86400)，输入仅 2048-dim | 信息瓶颈极端，应改用渐进上采样 |
| P2-5 | `slow_branch.py` | KeyframeHead 和 SceneTextHead 架构完全相同 | 可能学到高度冗余的映射 |
| P2-6 | `gated_fusion.py` | Sigmoid gates 无竞争（非 softmax），alpha_brain≈0.94 主导 | 考虑 softmax 或去掉 alpha_brain |
| P2-7 | `data_video.py` | ucg_rate=0.1 在 branch_pretrain 阶段也生效 | 应设为 0 |
| P2-8 | `data_video.py` | 视频帧无时序/空间数据增强 | 过拟合风险 |
| P2-9 | `eeg_encoder_custom.py` | 5 trial 的 64 通道直接展平为 320，丢失 trial 结构 | 考虑 trial attention pooling |
| P2-10 | 配置文件 | legacy 分类 head 参数仍在 model.yaml 中 | 混淆维护者 |
| P2-11 | 配置文件 | `lambda_dyn: 1.0` 在 model.yaml 中会覆盖代码默认值 0.1 | 死配置但可能引发混淆 |
| P2-12 | `loss.py` | loss breakdown 通过 `_last_loss_breakdown` 隐式属性传递 | 应改为正式返回值 |
| P2-13 | `sf_losses.py` | L_struct 的 MSE 包含对角线（恒为 1.0 vs 1.0 = 0），稀释有效梯度 | 只计算上三角非对角线 |
| P2-14 | `lora.py` | LoRA down weight init std=1/rank=1/128，对高 rank 偏小 | 考虑 std=1/sqrt(rank) |
| P2-15 | 各 yaml | weight_decay 应用于所有参数含 bias 和 norm | 标准做法排除 bias/norm |
| P2-16 | `get_metric.py` | `pred[:33]` 硬编码截断，帧数不匹配时会出错 | 动态对齐帧数 |
| P2-17 | 所有日志 | 无 gradient norm 记录 | 严重限制诊断能力 |
| P2-18 | `get_metric.py` | 缺少 brain-specific 评估指标 | 无法判断 conditioning 是否起作用 |
| P2-19 | 日志 | bitsandbytes CUDA 检测失败，缺少 xformers | 性能损失 15-30% |
| P2-20 | `sample_brain_va.py` | sampling_num_frames 未定义，推理脚本可能无法运行 | 显式定义参数 |

---

## 6. 方法设计 vs 实际实现不一致清单

| 设计文档规范 | 代码实际 | 一致？ | 影响 |
|---|---|---|---|
| L_dyn 用 2-class 分类 (static/dynamic) | 数据管道映射为 3-class `gt_dynamics_class`，loss 查找 `gt_dyn_label_2class`，**永远不匹配** | **断路** | P0 — 辅助 loss 完全失效 |
| `L_fast = lambda_distill * (L_cls + L_spatial)` | 代码用两个独立 lambda (lambda_cls, lambda_spatial) | **不一致** | 低 — 当前两个都设为相同值 |
| `lambda_temporal * (lambda_delta * L_delta + lambda_abs * L_abs)` | 代码无外层 lambda_temporal，直接用平级 lambda | **不一致** | 低 — 功能等价但增加配置错误风险 |
| L_guide = L_gk + L_gt + **L_gm** | `GuidanceLoss.forward()` 只算 L_gk + L_gt，**L_gm 从未计算** | **缺失** | P1 — motion guidance 无监督 |
| Keyframe 用第 1 帧 (Target 提取规范) | `extract_supervision_targets.py` 用**中间帧** (`T // 2`) | **不一致** | 中 — keyframe 与 delta 基准帧不同 |
| LoRA rank=64, alpha=64 (CONFIG_TEMPLATE) | 实际 rank=128, **alpha 未设置 (默认=1)** | **严重** | P0 — scaling=1/128 |
| SF loss 权重由 lambda 精确控制 | **动态缩放**将 sf_total 强制拉到 diff_loss 量级 | **未文档化** | P0 — 所有 lambda 失效 |
| CONFIG_TEMPLATE: `gating_hidden_dim: 1024` | model.yaml: hidden_dim=2048 | 有意变更 | 无影响 |
| L_aud (auditory context loss) | 未实现，lambda_aud=0 | 有意跳过 | 无影响 |
| Fast Branch 原始 4-head 设计 | 已替换为 P0蒸馏 + P1时序 | 有意变更 | 无影响 |

---

## 7. 性能瓶颈假设排序

按"最值得先排查"的顺序：

### 假设 1: LoRA scaling=1/128 导致 DiT 无法适配 brain conditioning
- **类型**: 实现 bug / 配置错误
- **依据**: Stage 3 所有 loss 960 iter 完全平坦；`lora_alpha` 默认值=1 代码确认；CogVideoX 标准配置用 alpha=rank
- **验证方式**: 比较 iter 0 和 iter 500 的 LoRA 权重差异（如果差异接近零则确认）
- **预期收益**: 修复后 Stage 3 的 L_diff 应能显著下降

### 假设 2: 动态缩放破坏了 loss 权重体系
- **类型**: 实现 bug / 设计缺陷
- **依据**: 代码确认 sf_total 被强制缩放到 diff_loss 量级；这使所有 lambda 调参无效；Stage 2 中被冻结模块的 loss 值稀释了可训练 loss
- **验证方式**: 添加日志记录动态缩放前后的 sf_total 和 scale 因子
- **预期收益**: 修复后 loss 权重体系恢复，各阶段的 loss 平衡可控

### 假设 3: 输入数据未归一化导致 encoder 学习不稳定
- **类型**: 数据问题
- **依据**: 代码确认无归一化逻辑；神经影像文献几乎都做 z-score
- **验证方式**: 检查 .npy 文件的数据分布（mean/std/range），如果 raw 则添加归一化后对比 overfit 速度
- **预期收益**: 如果数据确实未预处理，归一化后 encoder 学习效率可能大幅提升

### 假设 4: Guidance 信号被 brain_latent 残差主导
- **类型**: 方法本身瓶颈
- **依据**: alpha_brain=0.94 使 context ≈ 1.94*z_b + 小扰动；guidance 只是加法广播无空间特异性
- **验证方式**: 消融实验——去掉 alpha_brain 残差，对比视频生成质量
- **预期收益**: guidance 信号能更有效地控制视频生成的语义细节

### 假设 5: 226 tokens 超出 DiT 预训练分布 + gradient_clipping 过紧
- **类型**: 训练策略问题
- **依据**: CogVideoX 预训练 text tokens ≤77，现在 226 tokens；gradient_clipping=0.1 对 1.1B 参数模型过激进
- **验证方式**: (1) 将 brain tokens 降到 64 个；(2) 将 gradient_clipping 提高到 1.0
- **预期收益**: DiT 更容易适配，训练更稳定

---

## 8. 最优先的优化路线

### 修改 1: 修复 LoRA alpha（预期收益：极高）

```yaml
# configs/sf_v1/cinebrain_sf_v1_model.yaml
lora_config:
  target: sat.model.finetune.lora2.LoraMixin
  params:
    r: 128
    lora_alpha: 128  # ← 添加这一行
```

**验证方式**: 修复后重新跑 Stage 3 overfit test（1 sample, 200 iter），L_diff 应能快速下降。

### 修改 2: 移除动态缩放，使用固定权重（预期收益：高）

```python
# sgm/modules/diffusionmodules/loss.py, 行 320-323
# 删除以下代码:
# if sf_total.requires_grad:
#     scale = diff_loss.detach() / (sf_total.detach() + 1e-8)
#     sf_total = sf_total * scale

# 替换为:
lambda_sf = self.cfg.get("lambda_sf", 0.1)  # 可配置的固定权重
```

**验证方式**: 打印 diff_loss 和 sf_total 的量级，确认比例合理。

### 修改 3: 提高 gradient_clipping 到 1.0 + 添加 warmup（预期收益：中高）

```yaml
# configs/sf_v1/sf_v1_stage3_joint.yaml
deepspeed:
  gradient_clipping: 1.0  # 从 0.1 改为 1.0
```

同时在训练脚本中添加 warmup 调度。

### 修改 4: 修复 gt_dyn_label_2class 键名（预期收益：中）

```python
# data_video.py, _name_map 中添加:
def _transform_targets(self, sf_targets):
    # 3-class → 2-class 映射
    if "gt_dynamics_class" in sf_targets:
        dyn3 = sf_targets["gt_dynamics_class"]
        sf_targets["gt_dyn_label_2class"] = (dyn3 > 0).long()  # 0=static, 1/2=dynamic
    return sf_targets
```

### 修改 5: 验证并添加 fMRI/EEG 归一化（预期收益：可能很高）

```python
# data_video.py, get_item_func() 中:
# 加载 fMRI 后:
fmri = (fmri - fmri.mean(dim=-1, keepdim=True)) / (fmri.std(dim=-1, keepdim=True) + 1e-6)
# 加载 EEG 后:
eeg = (eeg - eeg.mean(dim=-1, keepdim=True)) / (eeg.std(dim=-1, keepdim=True) + 1e-6)
```

**先验证**: 用 `np.load(fmri_path).mean(), np.load(fmri_path).std()` 检查 raw 数据分布。

---

## 9. 可直接改动的补丁建议

### Patch A: LoRA alpha 一行修复

```yaml
# configs/sf_v1/cinebrain_sf_v1_model.yaml, lora_config.params 下添加:
lora_alpha: 128
```

### Patch B: 删除动态缩放

```python
# sgm/modules/diffusionmodules/loss.py, 删除 320-323 行，替换为:
lambda_sf = self.cfg.get("lambda_sf", 0.1)
sf_total = sf_total * lambda_sf
```

### Patch C: gradient_clipping

```yaml
# 所有 stage yaml 中:
gradient_clipping: 1.0
```

### Patch D: gt_dyn_label_2class 修复

```python
# data_video.py, get_item_func() 中 sf_targets 处理后添加:
if "gt_dynamics_class" in sf_targets:
    sf_targets["gt_dyn_label_2class"] = (sf_targets["gt_dynamics_class"] > 0).long()
```

### Patch E: 各阶段只计算可训练 loss

```python
# sgm/modules/diffusionmodules/loss.py, __call__() 中:
# Stage 2 (fusion): 跳过 L_fast 和 L_slow 计算
if self.training_stage == "fusion":
    # 只计算 L_guide，不计算冻结模块的 loss
    pass  # 具体实现需根据代码结构调整
```

### Patch F: 添加 gradient norm 日志

```python
# train_video_fmri.py 中，loss.backward() 后添加:
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))
log_dict["grad_norm"] = total_norm.item()
```

---

## 附录: 跨阶段问题传播链

```
Stage 1 (Fast Branch):
  ├── L_align 恒为 0 → 编码器无跨模态对齐
  ├── L_dyn 不计算 → dynamics head 未训练
  ├── fMRI/EEG 可能未归一化 → encoder 学习不稳定
  └── P1 temporal 遗忘 P0 蒸馏 → distill_spatial 暴涨 4x
       │
       ▼
Stage 2 (Fusion):
  ├── 动态缩放 + frozen loss 稀释 → L_guide 几乎不降 (<1%)
  ├── alpha_mot 动态分化消失 → motion guidance 形同虚设
  └── Guidance 被 brain_latent 主导 → 精细 guidance 被稀释
       │
       ▼
Stage 3 (Joint):
  ├── LoRA scaling=1/128 → DiT 几乎不更新 ← 最致命
  ├── 动态缩放 → lambda 调参全部无效
  ├── gradient_clipping=0.1 → 残余梯度被截断
  ├── 无 warmup → 训练起步不稳定
  └── 960 iter 所有 loss 完全平坦 → 训练可能浪费
```

**结论**: 问题从 Stage 1 开始积累，到 Stage 3 叠加了 LoRA scaling 问题后彻底导致训练失效。**修复后需要从 Stage 1 重新训练**，但优先修复 Patch A+B+C 后先在 Stage 3 overfit test 上验证效果。
