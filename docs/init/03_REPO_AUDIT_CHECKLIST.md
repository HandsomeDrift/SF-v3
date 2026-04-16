# Repo Audit Checklist — 必须先做，后编码

Claude 接手源码后，**第一步不是写代码，而是输出仓库审计报告**。  
请严格执行以下检查，并把结果写成一份 `repo_audit_report.md`。

## A. 仓库结构识别
请列出并简述以下目录/文件（如果存在）：
- dataset / data / dataloader
- models / modules / networks
- trainer / engine / runner
- configs / yaml / hydra
- scripts / train / eval / infer
- decoder / diffusion / CogVideoX / LoRA
- losses / metrics
- preprocess / ROI / EEG pipeline

## B. 数据流定位
请回答：
1. fMRI tensor 在哪里进入模型？
2. EEG tensor 在哪里进入模型？
3. visual ROI / auditory ROI 是否已分开？
4. 视频 latent 是在哪里构造的？
5. text embedding 是在哪里提取或加载的？
6. 当前 fused brain latent 是在哪里形成的？
7. 当前 decoder 的条件接口长什么样？

## C. 现有模块映射
请把现有 CineSync 代码映射成下面这些逻辑块：
- Multi-Modal Fusion Encoder
- Neuro Latent Decoder
- contrastive alignment
- diffusion denoising
- ROI preprocessing
- metric computation

## D. 可复用模块识别
请标记哪些模块可以直接复用：
- fMRI encoder
- EEG encoder
- CogVideoX/NLD
- dataloader
- metric code
- trainer loop
- LoRA hooks
- logging / checkpointing

## E. 新增模块挂接位点
请给出最合理的新增文件位置和挂接点：
- Slow branch
- Fast branch
- Gated fusion
- Multi-guidance decoder adapter
- new losses
- new config entries

## F. 配置系统检查
请确认：
- 现有配置管理方式（Hydra / yaml / argparse / custom）
- 如何注册新模型名
- 如何新增 loss 权重
- 如何开关 keyframe / motion / text guidance
- 如何开关 auditory ROI

## G. 最终输出格式
Claude 必须先输出以下内容后再写代码：
1. 仓库结构摘要
2. 现有 CineSync 数据流摘要
3. 新框架挂接方案
4. “最小侵入实现计划”
5. 风险点列表（例如 shape、显存、训练时序、数据预处理不一致）