# Project Brief — CineBrain-SF v1

## 1. 项目目标
在 **CineBrain / CineSync 源代码** 基础上，实现一个新的多模态脑信号到视频重建框架 **CineBrain-SF v1**。

SF = Slow-Fast，表示：
- **Slow branch（fMRI 主导）**：负责语义、结构、关键帧、场景关系、视听上下文
- **Fast branch（EEG 主导）**：负责运动、动态模式、时间一致性、转场节律

目标不是简单做一个更大的 CineSync，而是把现有统一 latent 路线升级为：
**显式 slow/fast 分工 + 多条件视频扩散解码**。

## 2. 核心研究命题
在自然视听叙事条件下，fMRI 与 EEG 是否分别编码了视频重建中不同时间尺度、不同信息类型的内容；
如果是，怎样把这种分工显式写进生成框架，而不是只学一个统一脑潜变量。

## 3. 与现有 CineSync 的差异
现有 CineSync：
- 双 Transformer 编码 fMRI / EEG
- 对齐到视觉/文本语义空间
- 融合为统一 brain latent
- 把 brain latent 送入 CogVideoX 风格 decoder

CineBrain-SF v1：
- 保留“分开编码再融合”的总体原则
- 新增 **Slow Semantic-Structure Branch**
- 新增 **Fast Motion-Dynamics Branch**
- 新增 **Cross-Modal Gated Fusion**
- 把 decoder 升级为 **Multi-Guidance Neuro-Latent Decoder**
- 不再只依赖一个 unified brain latent，而是联合：
  - keyframe guidance
  - scene-text guidance
  - motion guidance
  - fused brain latent guidance

## 4. 最小创新主张
第一版论文/实现不追求包揽所有问题，先聚焦这一个主张：

> 在 CineBrain 上，显式 slow-fast role assignment 是否优于 unified multimodal latent fusion 用于连续视频重建？

## 5. 第一版不强求的内容
以下内容可以作为后续扩展，不要求在 v1 首次实现里全部完成：
- 全量 cross-subject 训练
- 复杂 audiovisual joint generation
- 高分辨率 dense optical flow map 生成
- 图像/视频统一 task-conditioned 生成
- 复杂 caption autoregressive decoder

## 6. 第一版必须实现的内容
必须交付：
1. 基于现有仓库的 **可运行训练/验证代码**
2. 新模型的模块化挂接，而不是另起炉灶
3. 统一配置管理
4. 至少以下实验：
   - CineSync baseline 复现或对齐
   - unified latent vs slow-fast 双分支
   - slow only / fast only / full
   - motion guidance 消融
   - auditory ROI 消融
5. 中间变量的可视化或离线导出：
   - predicted keyframe
   - scene embedding / caption proxy
   - dynamic score
   - motion token / flow token（至少一种）

## 7. 编码总原则
- 优先复用现有 dataset / trainer / decoder / config / logging 体系
- 不要新建平行仓库
- 不要把所有逻辑写死在一个文件里
- 所有新增模块都要能单独开关
- 所有 loss、head、guidance 都要能在 config 中开关和调权重