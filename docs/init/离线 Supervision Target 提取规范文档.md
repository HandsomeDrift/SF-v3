# 离线 Supervision Target 提取规范文档

**项目：CineBrain-SF v1**
**用途：提供给 Claude 直接实现**

## 0. 任务目标

请在 **现有 CineBrain / CineSync 源代码** 基础上，实现一套**离线 supervision target 提取与缓存流程**，为后续 `CineBrain-SF v1` 训练提供稳定、可复现、低开销的训练目标。

本任务的核心要求是：

1. **不要在训练时在线提取 supervision targets**
2. 所有 target 必须基于 **clip-level 索引** 对齐
3. 所有 target 必须支持 **按 subject / run 分片缓存**
4. 所有 target 必须带有 **完整 metadata 和版本信息**
5. 尽量复用现有仓库中的数据索引、预处理和视频读取逻辑
6. 不要新建平行数据体系，优先兼容现有 dataloader / config / path 风格

------

## 1. 总体设计原则

### 1.1 提取目标分为两类

第一类是 **生成器友好目标**：

- keyframe VAE latent
- structure latent
- motion token / flow target

第二类是 **可解释中间变量目标**：

- keyframe image embedding
- scene / fusion text embedding
- dynamic score / fast-slow label

### 1.2 统一目标空间

请尽量避免为不同 head 混用过多不兼容的 embedding family。

推荐原则：

- **图像/文本语义空间**：统一用一套 image-text encoder
  - 首选：SigLIP family
  - 如果仓库已有深度绑定的 CLIP-family，则优先复用现有实现
- **结构空间**：统一使用最终 decoder 对应的 VAE latent space
- **运动空间**：统一使用 RAFT optical flow 及其派生 token / stats

### 1.3 以最小侵入方式落地

如果现有源码已有：

- 文本描述文件
- 预提取 video latent
- 现成的视觉 / 文本 encoder 包装器
- 现成的 clip 索引

请优先复用，而不是重新造一套。

------

## 2. 要提取的 supervision targets

请为每个 **4-second clip** 提取并缓存以下 target。

## 2.1 Keyframe 相关

每个 clip 选择一个固定 keyframe。

### 推荐默认方案

- 优先使用 **第 1 帧** 作为 keyframe
  原因：最容易与 first-frame / ControlNet / motion-conditioned generation 对齐

### 需要缓存的字段

- `keyframe_rgb`
  原始关键帧图像，建议可选缓存，不一定必须长期保留
- `keyframe_img_emb`
  关键帧图像语义 embedding
- `keyframe_vae_latent`
  关键帧在 decoder 对应 VAE 空间中的 latent

### 推荐模型

- `keyframe_img_emb`
  - 首选：SigLIP image encoder
  - 备选：现有代码已绑定的 CLIP-family image encoder
- `keyframe_vae_latent`
  - 必须使用最终 video decoder 对应的 VAE encoder
  - 若当前主 decoder 为 CogVideoX，则使用与其匹配的 VAE

------

## 2.2 Scene / Text 相关

每个 clip 需要一组文本语义监督。

### 需要缓存的字段

- `video_caption_raw`
- `audio_transcript_raw`
- `fusion_text_raw`
- `scene_text_emb`

### 推荐生成方式

- `video_caption_raw`
  - 使用 Qwen2.5-VL 对 clip 或 keyframe 生成简洁场景描述
- `audio_transcript_raw`
  - 使用 Whisper-large-v3 对同步音频片段生成转写
- `fusion_text_raw`
  - 默认用简单拼接模板构造，例如：

```text
Visual scene: {video_caption}. Spoken/audio context: {audio_transcript}.
```

- `scene_text_emb`
  - 使用与 `keyframe_img_emb` 同一族的 text encoder 编码
  - 即如果图像用 SigLIP，则文本也优先用 SigLIP text encoder
  - 如果仓库已有 CLIP-family text encoder 深度绑定，则复用之

### 要求

- 文本生成过程必须支持缓存结果，不能每次训练重新调用大模型
- 若已存在 CineBrain 论文附带或仓库预生成文本，请优先复用并记录来源

------

## 2.3 Structure 相关

structure target 不能等同于 scene semantics。

### 需要缓存的字段

- `structure_latent`

### 推荐定义

- 使用 keyframe 或第 1 帧在 decoder VAE 空间中的 latent
- 若条件允许，可额外缓存低层 pyramid 版本，但 v1 不是必须

### 推荐模型

- 与 decoder 完全一致的 VAE encoder

### 强约束

- 不要用 SigLIP / CLIP embedding 代替 structure latent
- structure target 必须贴近生成器空间，而不是高层语义空间

------

## 2.4 Motion / Flow 相关

这是 fast branch 的核心 supervision 之一。

### 需要缓存的字段

至少缓存以下之一，推荐全做：

- `flow_dense` 或 `flow_coarse`
- `flow_mag`
- `flow_token`
- `flow_stats`

### 推荐模型

- RAFT optical flow

### 推荐提取方案

1. 对 clip 内相邻帧提取 flow
2. 再做以下派生处理：
   - 平均 flow magnitude
   - 方向统计
   - patch pooling
   - 可选 k-means / VQ tokenization

### v1 推荐最小集合

至少缓存：

- `flow_coarse`
- `flow_mag`
- `flow_token`

### 说明

- 若 dense flow 占空间太大，可优先保存 coarse flow + token
- 但尽量保留能够恢复 motion metric 的信息

------

## 2.5 Dynamics 相关

这是 cheap but useful 的 target。

### 需要缓存的字段

- `ofs_score`
- `dyn_label`
- 可选：`motion_intensity_bin`

### 推荐定义

从 optical flow 统计一个 OFS（optical flow score），再根据阈值生成：

- `dyn_label ∈ {slow, fast}`

### 要求

- 阈值计算方式必须写入 metadata
- 如果做数据集内分位数划分，也必须记录版本

------

## 3. clip-level 索引规范

所有 target 必须严格绑定到统一的 clip 索引。

## 3.1 每个样本至少包含以下主键字段

- `subject_id`
- `episode_id`
- `run_id`
- `clip_id`
- `split`
- `start_frame`
- `end_frame`
- `fmri_index`
- `eeg_index`
- `audio_index`（如适用）

## 3.2 统一主键

建议使用：

```text
{subject_id}__{run_id}__{clip_id}
```

作为唯一字符串 ID，例如：

```text
sub03__run12__clip00458
```

所有离线提取出的 target 都必须能通过这个 ID 一一对应回原始数据样本。

------

## 4. 缓存格式规范

## 4.1 分片策略

不要一个 clip 一个小文件。
请按 **subject / run** 做 shard。

### 推荐目录结构

```text
supervision_cache/
  version_v1/
    metadata/
      global_index.parquet
      extraction_config.yaml
      version_info.json
    sub01/
      run01_targets.pt
      run02_targets.pt
      ...
    sub02/
      run01_targets.pt
      ...
```

## 4.2 每个 shard 建议包含

以 `runXX_targets.pt` 为例，内部应保存字典：

```python
{
  "clip_ids": [...],
  "keyframe_img_emb": Tensor[N, D1],
  "keyframe_vae_latent": Tensor[N, ...],
  "scene_text_emb": Tensor[N, D2],
  "structure_latent": Tensor[N, ...],
  "flow_token": Tensor[N, ...],
  "flow_mag": Tensor[N, ...],
  "ofs_score": Tensor[N],
  "dyn_label": Tensor[N],
  "meta": {...}
}
```

### 可选不长期缓存

如果磁盘压力大，这些可以只保留在 metadata 或单独压缩存储：

- `keyframe_rgb`
- `video_caption_raw`
- `audio_transcript_raw`
- `fusion_text_raw`

但建议至少在初版保留 raw text，方便排查和后续分析。

------

## 5. metadata 与版本管理

请务必为每套 target 保存完整版本信息。

## 5.1 必须记录的 version_info

- `target_version`
- `keyframe_rule`
- `image_encoder_name`
- `text_encoder_name`
- `caption_model_name`
- `asr_model_name`
- `vae_name`
- `flow_model_name`
- `flow_tokenizer_name`
- `ofs_threshold_rule`
- `image_resolution`
- `video_fps_used`
- `clip_seconds`
- `creation_time`
- `git_commit`（如果方便）

## 5.2 extraction_config.yaml

请把本次提取的所有核心参数写入一个 yaml，包括：

- 模型名
- batch size
- frame sampling 规则
- keyframe 规则
- tokenization 参数
- OFS 阈值策略
- 是否复用已有文本
- 是否保存 raw RGB / raw text

------

## 6. 推荐模型与默认选择

如果源码没有现成强绑定，建议默认使用以下组合。

### 6.1 图像 / 文本语义

- image encoder: **SigLIP**
- text encoder: **SigLIP text encoder**
- video caption model: **Qwen2.5-VL**
- audio ASR model: **Whisper-large-v3**

### 6.2 结构 latent

- 使用 **当前 decoder 对应的 VAE encoder**
- 如果当前 decoder 为 CogVideoX，则直接用其 VAE

### 6.3 运动 / 动态

- flow extractor: **RAFT**
- flow tokenization: patch pooling + optional k-means
- dynamics label: OFS-based fast/slow

## 6.4 若仓库已有等价模块

如果仓库已有深度绑定模型：

- 优先复用现有 wrapper / checkpoint / preprocessing
- 不强制改成 SigLIP
- 但必须保证 image/text 使用同一语义空间

------

## 7. 提取流程要求

请实现一个独立可复用的离线提取脚本或 pipeline。

## 7.1 推荐执行顺序

1. 加载全局 clip 索引
2. 遍历 subject / run
3. 对每个 clip：
   - 定位视频帧
   - 选 keyframe
   - 提取 keyframe image embedding
   - 提取 keyframe / structure VAE latent
   - 生成/加载视频 caption
   - 生成/加载音频 transcript
   - 编码 fusion text embedding
   - 提取相邻帧 flow
   - 生成 flow token / flow stats
   - 计算 OFS 与 dyn_label
4. 写入 shard
5. 更新 global metadata index
6. 记录失败样本与异常日志

## 7.2 要求支持断点续跑

如果某个 subject/run 已完成，脚本应支持：

- 跳过已存在 shard
- 或 `--force_rebuild` 重建

------

## 8. 质量检查与验收标准

实现完成后，请至少做以下检查。

## 8.1 完整性检查

- 每个训练 clip 都有 target
- 每个验证 clip 都有 target
- shard 中 clip 数与 global index 一致
- 没有重复 clip_id
- 没有缺失 subject/run

## 8.2 shape 检查

请输出一个检查报告，至少包含：

- `keyframe_img_emb.shape`
- `scene_text_emb.shape`
- `keyframe_vae_latent.shape`
- `structure_latent.shape`
- `flow_token.shape`
- `flow_mag.shape`
- `dyn_label.shape`

## 8.3 语义合理性 spot check

随机抽样若干 clip，输出：

- keyframe 图
- video caption
- audio transcript
- fusion text
- flow 可视化
- dyn_label

要求人工可以快速判断 target 是否明显错位。

## 8.4 存储规模检查

请报告：

- 每个 run shard 大小
- 每个 subject 总大小
- 全量缓存总大小
- 如果太大，给出压缩或分级缓存建议

------

## 9. 与训练代码的接口要求

请额外提供一个最小的 target loader 接口，供训练阶段直接使用。

## 9.1 dataloader 输出应支持

在训练样本中可直接取到：

```python
sample = {
  "fmri": ...,
  "eeg": ...,
  "video": ...,
  "keyframe_img_emb": ...,
  "scene_text_emb": ...,
  "keyframe_vae_latent": ...,
  "structure_latent": ...,
  "flow_token": ...,
  "flow_mag": ...,
  "ofs_score": ...,
  "dyn_label": ...
}
```

## 9.2 开关控制

在 config 中必须支持开关：

- `use_keyframe_target`
- `use_scene_text_target`
- `use_structure_target`
- `use_motion_target`
- `use_dynamic_target`

这样后续做消融时不需要重新改 dataloader。

------

## 10. 需要 Claude 最终交付的内容

Claude 完成该任务后，必须交付：

1. **新增/修改文件清单**
2. **离线 target 提取脚本**
3. **配置文件示例**
4. **缓存目录结构说明**
5. **metadata 字段说明**
6. **一个 shape/完整性检查脚本**
7. **一个最小 dataloader 接口示例**
8. **运行命令示例**
9. **已知风险与后续建议**

------

## 11. 给 Claude 的执行提醒

- 优先复用现有 CineBrain 数据索引和视频读取逻辑
- 不要在训练阶段在线调用 Qwen / Whisper / SigLIP / RAFT
- 不要把 structure target 直接偷懒设成 image embedding
- 不要让不同 target 使用互不兼容且未记录的 embedding family
- 不要生成“一堆零散小文件”
- 所有目标必须可通过 `clip_id` 回溯到原始数据样本

------

## 12. 最终执行目标

请实现一套稳定、可复现、可版本管理的离线 supervision cache，使后续 `CineBrain-SF v1` 训练阶段可以直接读取以下目标：

- keyframe image embedding
- scene / fusion text embedding
- keyframe / structure VAE latent
- motion token / flow target
- dynamics label / score

并确保该缓存体系可直接支持：

- slow branch 训练
- fast branch 训练
- multi-guidance decoder 训练
- 后续消融实验