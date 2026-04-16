# CineBrain-SF v1 — 创新点候选方案

**Date**: 2026-04-16
**Last Updated**: 2026-04-16 (补充 CogVideoX 架构可行性分析)
**Status**: 待探索，后续逐个深入设计和验证

---

## 总体思路

当前框架的核心空白在三个环节：

| 环节 | 现状 | 缺的东西 |
|------|------|---------|
| **分解** (怎么分) | 人工指定 fMRI=慢, EEG=快 | 没有原则性的分解机制 |
| **利用** (怎么用) | 静态 alpha 加权注入 DiT | 没有跟扩散过程的内在联系 |
| **约束** (怎么保证分开) | cosine ≈ 0 被动观察 | 没有显式的时间尺度分离训练目标 |

最有深度的创新应该从这三个空白中诞生。

---

## CogVideoX-5B 架构可行性分析 (2026-04-16 研究结论)

### 关键架构发现

1. **Joint Self-Attention (非标准 Cross-Attention)**
   - CogVideoX-5B 将 brain tokens (226个, 4096→3072 投影) 和 image patches (17550个) **拼接成一个序列**
   - 42 层 self-attention 中两部分互相 attend
   - AdaLN 用 timestep embedding 分别调制 text tokens 和 image tokens 的 shift/scale/gate

2. **Context 只计算一次**
   - `SFBrainEmbedder.forward()` 在采样循环前调用一次，输出 `context (B, 226, 4096)`
   - 存入 `cond["crossattn"]`，50 步去噪中原封不动重复使用
   - **→ 如果要做 timestep-aware guidance，需要在循环内拦截并修改 context**

3. **采样循环可干预**
   - `VPSDEDPMPP2MSampler.__call__` 是普通 Python for 循环
   - 已有两个步内干预先例：
     - **DynamicCFG**: CFG scale 随步数变化 `1 + s*(1-cos(π*(i/N)^exp))/2`
     - **Alpha-Guidance renoise**: 在指定步骤对已去噪结果重新加噪
   - 添加 context 修改只需在循环体内插入代码

4. **Latent 帧维度**
   - shape: `(B, T=13, C=16, H=60, W=90)` — 33 video frames → VAE 4x 时间压缩 → 13 latent frames
   - DANA 已做 per-frame 噪声修改 (`flow_traj → beta → per-frame noise mixing`)，证明帧级操作可行

5. **Timestep 在 DiT 内部的使用方式**
   - timestep → sinusoidal embedding → MLP → `emb (B, 512)`
   - emb 通过 AdaLN 在每一层计算 12 个调制参数 (shift/scale/gate)
   - brain tokens 和 image tokens 各自有独立的 AdaLN 调制
   - **cross-attention 层不直接接收 t**，但 t 通过 AdaLN 间接影响所有层

### 代码位置速查

| 功能 | 文件 | 关键行号 |
|------|------|---------|
| 采样入口 & context 一次性计算 | `sample_brain_va.py` | 187-230 |
| VPSDEDPMPP2MSampler 去噪循环 | `sgm/.../sampling.py` | 726-773 |
| DynamicCFG (已有 timestep-aware scale) | `sgm/.../guiders.py` | 56-74 |
| Alpha-Guidance renoise | `sgm/.../sampling.py` | 741-745 |
| DiT forward (timestep → AdaLN) | `dit_video_concat_fmri.py` | 748-773 |
| AdaLN 调制 (12 个参数) | `dit_video_concat_fmri.py` | 449-519 |
| brain tokens + image patches 拼接 | `dit_video_concat_fmri.py` | ImagePatchEmbeddingMixin |
| SFBrainEmbedder context 生成 | `sgm/.../sf_embedder.py` | 200-255 |
| MultiGuidanceAdapter alpha 加权 | `sgm/.../multi_guidance.py` | 103-136 |
| DANA per-frame noise mixing | `diffusion_video_brain.py` | 327-344 |

---

## 方向一：扩散时间步自适应引导 (Timestep-Aware Slow-Fast Guidance)

**解决的问题**: "怎么用" — 将慢快变量与扩散过程的 coarse-to-fine 结构对齐

### 核心观察

扩散模型的去噪过程本身就有"先慢后快"的内在结构：
- **高噪声阶段 (t 大, 早期步)**: 全局语义规划 — "这段视频大概是什么场景"
- **低噪声阶段 (t 小, 后期步)**: 局部细节填充 — "运动/纹理/过渡怎么画"

但现有所有 brain-to-video 方法（包括我们）都用固定不变的 guidance 权重。这意味着在应该聚焦全局结构的早期步骤，运动 guidance 可能在"捣乱"；在应该聚焦动态细节的后期步骤，语义 guidance 的信号已经没用了。

### 提案

让 guidance 权重成为扩散 timestep t 的函数：

```
α_key(t), α_txt(t) → t 大时高，t 小时衰减    (慢变量: 先起主导)
α_mot(t)           → t 大时低，t 小时增强      (快变量: 后期接管)
α_brain(t)         → 全程稳定                  (基础 context)
```

### 三种实现路径 (经 CogVideoX 架构验证)

**路径 A — 推理时 schedule 插值 (零训练成本, ~20 行代码)**

```python
# 修改 VPSDEDPMPP2MSampler.__call__ 的循环
# 预计算两组 context: context_slow (高 α_key/α_txt) 和 context_fast (高 α_mot)
for i in range(num_sigmas - 1):
    t_ratio = i / (num_sigmas - 1)  # 0→1 (高噪→低噪)
    
    # cosine schedule: 先慢后快
    w_slow = math.cos(math.pi * t_ratio / 2)
    w_fast = math.sin(math.pi * t_ratio / 2)
    cond_t["crossattn"] = w_slow * context_slow + w_fast * context_fast
    
    x, old_denoised = self.sampler_step(..., cond=cond_t, uc=uc)
```

优点：不需要训练，立刻可以验证概念
类比：DynamicCFG 已经是同一模式的先例

**路径 B — 缓存中间结果 + 逐步重组 (零训练成本, 更精细)**

```python
# 在 SFBrainEmbedder.forward() 中缓存中间产物:
# z_b, g_key, g_txt, g_mot (MultiGuidanceAdapter 的输入)

# 采样循环中, 按 timestep 用不同 alpha 重新组合:
for i in range(num_sigmas - 1):
    t_ratio = i / (num_sigmas - 1)
    alpha_key_t = base_alpha_key * (1 - t_ratio)  # 早期高
    alpha_mot_t = base_alpha_mot * t_ratio          # 后期高
    
    context_t = z_b.clone()
    context_t += alpha_key_t * key_attn(z_b, g_key)
    context_t += alpha_mot_t * mot_attn(z_b, g_mot)
    ...
```

优点：精细控制每个通道，不需要预计算两组 context
代价：每步多算几次 cross-attention (~1% 额外耗时)

**路径 C — 可学习的 timestep-conditioned gating (需要训练)**

```python
# 修改 CrossModalGatedFusion.gate_net
# 输入: pooled_slow || pooled_fast || timestep_embedding
# 输出: 4 × alpha(t)
class TimestepAwareGateNet(nn.Module):
    def forward(self, slow_feat, fast_feat, t_emb):
        pooled = cat(pool(slow_feat), pool(fast_feat), t_emb)
        return sigmoid(self.mlp(pooled))  # {α_key(t), α_txt(t), α_mot(t), α_brain(t)}
```

优点：端到端学习最优的 timestep-alpha mapping
代价：需要在训练中每步传入 t（当前 loss 计算已有 t，可直接传递）

### 推荐执行顺序

```
路径 A (1-2 天) → 验证概念是否有效 (FVD/EPE 对比)
   ↓ 如果有效
路径 B (2-3 天) → 精细化，找最优 schedule
   ↓ 如果值得投入训练
路径 C (1-2 周) → 端到端训练，学到可展示的 α(t) 曲线
```

### 创新价值

1. 据调研没有任何 brain-to-video 论文做过 timestep-aware guidance switching
2. 扩散模型的 coarse-to-fine 与慢-快变量的时间尺度天然对应
3. 可能解决 Gating 帕累托问题：不同 timestep 用不同权重，把"画质 vs 路由"的冲突在时间维度上分解
4. DynamicCFG 已验证了 timestep-dependent scale 的模式，我们扩展到 context composition

### 验证计划

- 对比 static alpha vs timestep-aware alpha 的 FVD/EPE
- 可视化学到的 α(t) 曲线：是否自发出现了"先慢后快"的模式
- 如果出现了，这就是论文核心图
- 分别在高动态和低动态 clip 子集上评估

### 评估

| 维度 | 评分 |
|------|------|
| 创新深度 | 高 |
| 实现难度 | **低** (路径 A/B 零训练成本) |
| Slow-Fast 故事契合度 | 极高 (慢-快 × 粗-细) |
| CogVideoX 可行性 | **已验证** |
| 独立发表价值 | 可作为核心贡献之一 |

### 潜在风险

- 路径 A/B 的解析式 schedule 可能不是最优的，但足以验证概念
- 路径 C 训练时需要在每个 diffusion step 都跑一次 gating，计算量增加但可控
- 如果慢快变量的最优切换点不是 monotonic 的（比如中间步骤需要两者共存），简单 schedule 可能不够

---

## 方向二：对比式时间尺度分离 (Contrastive Temporal Scale Separation)

**解决的问题**: "怎么保证分开" — 将 Slow-Fast 分治从被动观察变为显式训练目标

### 核心问题

当前 Fast/Slow cosine ≈ -0.04 是被动观察到的，不是训练目标。模型碰巧学到了正交表征，但：
- 不鲁棒：换数据集 / 训练 seed 可能就不正交了
- 不可解释：正交 ≠ "一个是慢变量一个是快变量"

### 提案

设计显式的**时间尺度对比 loss**：

```python
# 取时间相邻的两个 clip (clip_t 和 clip_{t+1}) 的特征
slow_t, slow_t1 = slow_branch(fmri_t), slow_branch(fmri_{t+1})
fast_t, fast_t1 = fast_branch(eeg_t),  fast_branch(eeg_{t+1})

# 慢变量: 相邻 clip 应该相似 (场景语义不会突变)
L_slow_stable = 1 - cosine(slow_t, slow_t1)

# 快变量: 相邻 clip 应该有差异 (运动/动态在变化)
L_fast_diverse = max(0, cosine(fast_t, fast_t1) - margin)

# 交叉约束: 慢变量不应包含快变量的信息 (反之亦然)
L_cross = |cosine(slow_t, fast_t1 - fast_t)|

L_temporal_contrast = L_slow_stable + L_fast_diverse + L_cross
```

### CogVideoX 相关性

此方向不涉及扩散模型修改，在 Branch 训练阶段 (Stage 1) 的 loss 函数层面实施。需要的改动：
- 数据加载：从单 clip 改为加载相邻 clip pairs
- loss.py：新增 `L_temporal_contrast` 计算
- 不影响推理流程

### 创新价值

1. 形式化了 "分治" 的概念：把直觉性的"fMRI 编码慢变量"变成可优化的数学目标
2. 有理论渊源但有原创性：受 Slow Feature Analysis (SFA, Wiskott 2002) 和 VICReg 启发，但应用到多模态脑信号时间尺度分离是全新的
3. 解决证据链问题：训练后 slow 特征跨时间稳定、fast 特征跨时间多变 = Slow-Fast 故事的最直接定量证据
4. 需要数据改动：加载相邻 clip pairs，非平凡但可行

### 验证计划

- 训练前后 slow/fast 特征的时间自相关函数对比
- slow 特征在相邻 clip 间的 cosine 分布 vs fast 特征分布 → 应明显分开
- 消融: 有/无 L_temporal_contrast 的下游 FVD/EPE 对比

### 评估

| 维度 | 评分 |
|------|------|
| 创新深度 | 高 |
| 实现难度 | 中 |
| Slow-Fast 故事契合度 | **极高** (直接形式化分治) |
| CogVideoX 相关性 | 不涉及 (训练端 loss) |
| 独立发表价值 | 可作为核心贡献之一 |

### 潜在风险

- 相邻 clip 的假设"语义不变但运动变"不总是成立（场景切换时语义也变）
  - 缓解: 根据 scene boundary 检测过滤切换点附近的 pair
- `L_fast_diverse` 的 margin 选择需要调参
- 数据加载改为 pair 可能降低训练吞吐量（但可以用 cache 缓解）

---

## 方向三：不确定性感知的动态融合 (Uncertainty-Aware Dynamic Fusion)

**解决的问题**: "怎么分" — 用不确定性替代人工 gating，实现原则性的融合

### 核心问题

当前 gating 是"信心盲"的——输出 α_mot=0.3，但不知道是"确定该用 0.3"还是"完全不确定随便给的"。脑信号天然嘈杂，不同样本各模态信号质量差异极大。

### 提案

让每个 Branch 输出预测不确定性，用不确定性驱动融合：

```python
# Slow Branch 输出均值和不确定性
z_key_mu, z_key_logvar = slow_branch.keyframe_head(fmri)  # (B, 1152) × 2
z_key = z_key_mu + eps * exp(0.5 * z_key_logvar)          # reparameterization

# Fast Branch 同理
temporal_mu, temporal_logvar = fast_branch.temporal_decoder(eeg)

# 精度加权融合 (precision weighting)
precision_slow = 1.0 / (exp(z_key_logvar).mean() + eps)
precision_fast = 1.0 / (exp(temporal_logvar).mean() + eps)

α_semantic = precision_slow / (precision_slow + precision_fast)
α_dynamic  = precision_fast / (precision_slow + precision_fast)
```

### CogVideoX 相关性

此方向在 encoder 侧修改，不涉及扩散模型。影响 context 的构成，但不改变传递方式。

### 创新价值

1. 贝叶斯理论基础：精度加权融合是贝叶斯多感官整合的经典框架
2. 跟神经科学叙事完美契合："大脑在融合多感官信息时，会自动根据各感官的可靠性动态调整权重"
3. 自然解决 gating 训练困难：不需要额外的 router loss，不确定性本身提供分配信号
4. 附带正则化效果：logvar 输出防止 branch 过度自信

### 验证计划

- 不确定性是否与实际预测误差相关 (calibration 分析)
- 高/低质量样本的不确定性分布
- 对比 uncertainty-weighted vs learned gating vs fixed weights

### 评估

| 维度 | 评分 |
|------|------|
| 创新深度 | 中-高 |
| 实现难度 | 中 |
| Slow-Fast 故事契合度 | 高 (贝叶斯 + 神经科学) |
| CogVideoX 相关性 | 不涉及 (encoder 端) |
| 独立发表价值 | 辅助贡献 |

### 潜在风险

- logvar 可能在训练初期不稳定，需要 KL annealing
- 精度加权替代 gating 后，模型可能退化为始终信任某一个 branch（需要 minimum uncertainty floor）

---

## 方向四：快变量引导的扩散轨迹整形 (Fast-Variable Trajectory Shaping)

**解决的问题**: "怎么用" — 将 Fast Branch 输出与扩散过程深度整合

### 核心观察

Fast Branch 预测了每帧的运动强度轨迹 `flow_traj_pred (B, T)`，但目前只在初始噪声 (DANA) 和 cross-attention context 中使用。扩散模型的去噪轨迹本身可以被整形。

### 提案与 CogVideoX 可行性

**方案 A: Per-frame guidance scale (推理时修改，安全)**

```python
# 修改 DynamicCFG，将 scale 从标量扩展为 (T,) tensor
# 高运动帧用更大的 brain guidance scale
s_per_frame = s_base + s_delta * normalized_flow_traj  # (B, T)

# 在 guider 的 output 计算中按帧维度 broadcast
eps_guided[:, t, ...] = eps_uncond[:, t, ...] + s_per_frame[:, t] * (eps_cond[:, t, ...] - eps_uncond[:, t, ...])
```

已验证可行: DANA 已经是 per-frame 操作的先例，latent shape `(B, T=13, C, H, W)` 支持帧维度索引。

**方案 B: Per-frame noise schedule (需要训练，风险较大)**

```python
beta_per_frame = base_beta * (1 + gamma * normalized_flow_traj)  # (B, T)
```

改变了训练目标分布，DiT 从未见过帧间噪声水平不一致的情况。风险较高。

**方案 C: Per-frame context weighting (路径 B 的自然延伸)**

结合方向一的路径 B，在每个去噪步对每帧使用不同的 α_mot：

```python
# 高运动帧: α_mot 更大 (需要更多动态引导)
# 低运动帧: α_key 更大 (静态场景靠语义)
alpha_mot_per_frame = base_alpha_mot * (1 + flow_traj_pred)  # (B, T)
```

这需要将 MultiGuidanceAdapter 的 alpha 从 `(B, 1)` 扩展到 `(B, T, 1)`。

### 创新价值

1. 把 Fast Branch 输出跟扩散过程深度整合，不再只是 context 维度
2. Per-frame adaptive generation 在视频扩散中本身是前沿
3. 直接利用 Fast Branch 独有的时序维度信息

### 评估

| 维度 | 评分 |
|------|------|
| 创新深度 | 中-高 |
| 实现难度 | 中 |
| Slow-Fast 故事契合度 | 高 (快变量深度利用) |
| CogVideoX 可行性 | **已验证** (方案 A/C 安全可行) |
| 独立发表价值 | 辅助贡献 |

### 潜在风险

- 方案 B 改变训练分布，可能需要从头训练
- per-frame 操作可能引入帧间不一致的伪影
- flow_traj_pred 本身精度有限 (Pearson ~0.36)，用不准确的信号引导可能适得其反

---

## 推荐组合与整体叙事

### 核心组合: ①+②

- **②** 解决 "怎么保证分开" (训练目标层面的创新)
- **①** 解决 "怎么用" (推理生成层面的创新)

**论文叙事**:
> "我们不仅在训练中通过对比式时间尺度分离显式分离了快慢变量（②），还在扩散生成过程中按时间步自适应地利用它们（①），使慢变量在高噪声阶段主导全局结构规划，快变量在低噪声阶段接管动态细节填充。"

### 扩展组合: ①+②+④

- **④** 的方案 A (per-frame guidance scale) 作为第三个贡献，将快变量的利用从 timestep 维度扩展到 frame 维度
- 叙事: "快变量在两个维度上引导生成——去噪维度 (timestep-aware) 和帧维度 (frame-adaptive)"

### 执行优先级

```
第一优先 (可立即验证):
  → 方向① 路径 A: 在推理循环中加 schedule, 零训练成本, 1-2 天可出结果

第二优先 (需要数据改动):
  → 方向②: 修改数据加载 + 新增 loss, Stage 1 重训, 1-2 周

第三优先 (精细化):
  → 方向① 路径 C: 训练可学习的 α(t), 与方向② 的重训合并
  → 方向④ 方案 A: 在验证方向① 有效的基础上顺手加入

可选 (独立):
  → 方向③: 不确定性融合, 独立于上述方向, 可并行探索
```

---

## 前置问题状态

| 问题 | 状态 | 结论 |
|------|------|------|
| CogVideoX 是否支持 per-timestep guidance? | **已解决** | 完全支持，采样循环可干预，DynamicCFG 是先例 |
| Context 能否逐步修改? | **已解决** | 可以，路径 A/B/C 三种方案均可行 |
| Gating network 能否扩展为 timestep-conditioned? | **已解决** | 可以，加 t_emb 输入即可 (路径 C) |
| 数据加载能否支持相邻 clip pairs? | **待验证** | 方向②需要，需查看 data_video.py 的实现 |
| VAE latent 帧维度与 flow_traj 对齐? | **已解决** | latent T=13, flow_traj T=T_out (配置可对齐) |
