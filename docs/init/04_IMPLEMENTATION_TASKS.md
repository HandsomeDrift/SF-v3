# Implementation Tasks — 分阶段编码任务

## Task 0. Repo audit
- 按 `03_REPO_AUDIT_CHECKLIST.md` 完成审计
- 输出 `repo_audit_report.md`

## Task 1. Baseline freeze
目标：
- 跑通现有 CineSync baseline
- 记录输入 shape、fused latent shape、decoder condition shape
- 产出 baseline config 与 baseline log

交付：
- baseline training command
- baseline eval command
- baseline shapes summary

## Task 2. Build Slow Branch
新增模块：
- `SlowBranch`
- `fMRIVisualEncoderWrapper`（可复用现有）
- `fMRIAuditoryEncoderWrapper`（可复用现有或分支化）
- `AudiovisualContextAdapter`
- `KeyframeHead`
- `SceneTextHead`
- `StructureHead`

要求：
- 每个 head 都支持开关
- 所有输出 shape 可打印
- 支持独立 forward 测试

## Task 3. Build Fast Branch
新增模块：
- `FastBranch`
- `EEGSpatialEncoderWrapper`
- `EEGTemporalEncoderWrapper`
- `DynamicsHead`
- `MotionHead`
- `TemporalCoherenceHead`

要求：
- 先实现 motion latent 版本
- flow token 作为 optional 子模式
- 支持单独训练 / 调试

## Task 4. Build Cross-Modal Gated Fusion
新增模块：
- `CrossModalGatedFusion`
- 输出 `alpha_key, alpha_txt, alpha_mot, alpha_brain`
- 输出 fused latent `z_b`

要求：
- 与原 CineSync late fusion 对齐接口
- 可退化成固定权重模式

## Task 5. Upgrade Decoder to Multi-Guidance
新增内容：
- decoder condition adapter
- keyframe guidance path
- text guidance path
- motion guidance path
- fused latent guidance path

要求：
- 不破坏原有 decoder 主干
- 支持仅 brain latent 模式回退
- 支持逐项 guidance 消融

## Task 6. Add losses
新增 loss：
- `loss_align.py`
- `loss_slow.py`
- `loss_fast.py`
- `loss_guidance.py`

要求：
- 全部权重放进 config
- 支持日志记录每项 loss

## Task 7. Stage-wise training
实现三阶段训练配置：
- branch pretraining
- fusion training
- joint decoding

要求：
- 同一 trainer 下用 config 切换
- 不要写三套独立训练脚本，除非现有框架强制要求

## Task 8. Eval and visualization
新增评估内容：
- keyframe similarity
- text/scene similarity
- dynamics accuracy
- motion latent / flow metric
- guidance ablation logging

新增可视化：
- predicted keyframe
- dynamic score
- motion token map / flow token summary
- alpha weights per sample

## Task 9. Documentation
必须同步更新：
- model registry
- config example
- train/eval command
- tensor shape notes
- ablation command examples