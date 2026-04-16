# Prompt for Claude

你现在的任务不是从零重写一个新仓库，而是**基于现有 CineBrain / CineSync 源代码**，实现一个新的模型框架 **CineBrain-SF v1**。

## 你的工作目标
请在现有代码基础上，实现一个“slow-fast 分治”的多模态脑信号视频重建框架：

- **Slow branch（fMRI 主导）**：负责语义、结构、关键帧、场景关系、视听上下文
- **Fast branch（EEG 主导）**：负责运动、动态模式、时间一致性、转场节律
- **Fusion**：不做简单 early fusion，而是先保留分支编码，再做 gated fusion
- **Decoder**：保留现有 CogVideoX / NLD 主干，但从单一 brain latent 条件升级为 multi-guidance：
  - keyframe guidance
  - scene-text guidance
  - motion guidance
  - fused brain latent guidance

## 第一优先级要求
你必须优先满足以下原则：
1. **最小侵入**：优先复用现有数据流、trainer、decoder、config 体系
2. **不新建平行仓库**
3. **先做仓库审计，再写代码**
4. **所有新模块可单独开关**
5. **所有 loss 和 guidance 权重进 config**
6. **能跑 baseline，也能跑新模型**
7. **先保证可运行，再逐步扩展**

## 必须先完成的步骤
你开始编码前，必须先执行并输出：
- `03_REPO_AUDIT_CHECKLIST.md`
- 并生成一份 `repo_audit_report.md`

报告必须回答：
- 当前 CineSync 的 fMRI / EEG / fused latent / decoder condition 是怎么流动的
- 最合理的新增挂接点在哪
- 哪些模块可复用
- 哪些地方要新增文件
- 当前最大风险点是什么

**在给出审计报告之前，不要直接开始改代码。**

## 你要实现的模块
请按以下模块命名或做同义映射：
- `SlowBranch`
- `FastBranch`
- `CrossModalGatedFusion`
- `MultiGuidanceDecoderAdapter`
- `KeyframeHead`
- `SceneTextHead`
- `StructureHead`
- `DynamicsHead`
- `MotionHead`
- `TemporalCoherenceHead`

如果仓库已有相近模块，请优先复用并改造。

## 训练阶段
请支持三阶段训练：
1. Stage I: branch pretraining
2. Stage II: fusion training
3. Stage III: joint video decoding

## 最小实验要求
请至少支持以下实验配置：
- CineSync baseline
- slow only
- fast only
- full model
- w/o keyframe
- w/o text
- w/o structure
- w/o motion
- w/o auditory ROI
- brain latent only vs multi-guidance

## 你的输出格式
请按下面格式回复，不要省略：
1. 仓库审计报告
2. 实现计划（按文件）
3. 已修改/新增文件清单
4. 每个文件的核心改动说明
5. 配置示例
6. 训练命令
7. 验证命令
8. 当前还未完成的事项
9. 风险与下一步建议

## 重要提醒
如果源码结构和文档假设不一致，请：
- 保留源码命名风格
- 在报告中说明映射关系
- 不要强行套用文档中的示例路径

你的目标是：**让这个方案在 CineBrain 源码中尽快变成可训练、可消融、可扩展的第一版实现。**