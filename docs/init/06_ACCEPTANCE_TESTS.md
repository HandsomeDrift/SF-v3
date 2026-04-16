# Acceptance Tests — 最小验收标准

## A. Repo-level acceptance
- [ ] 不新建平行仓库
- [ ] 现有 baseline 仍可运行
- [ ] 所有新增模块都通过 import
- [ ] config 能切换 baseline / ours

## B. Shape-level acceptance
在一次 forward 中打印并确认：
- [ ] fMRI input shape
- [ ] EEG input shape
- [ ] slow branch outputs
- [ ] fast branch outputs
- [ ] fused latent shape
- [ ] each guidance shape
- [ ] decoder condition shape
- [ ] output video latent / video shape

## C. Function-level acceptance
- [ ] 只开 slow branch 能跑
- [ ] 只开 fast branch 能跑
- [ ] full model 能跑
- [ ] decoder 可切换：
  - brain-latent only
  - + keyframe
  - + text
  - + motion
  - full multi-guidance

## D. Training-level acceptance
- [ ] Stage I 能收敛若干 step
- [ ] Stage II 能收敛若干 step
- [ ] Stage III 能完成最小训练
- [ ] loss 日志含：
  - total
  - align
  - slow
  - fast
  - diffusion
  - guidance

## E. Metric-level acceptance
- [ ] 能输出主视频指标
- [ ] 能输出至少 2 个中间变量指标
- [ ] 能导出 keyframe 预测可视化
- [ ] 能导出 motion / dynamics 可视化

## F. Minimal performance acceptance
不是最终论文标准，只是工程门槛：
- [ ] full model 不应显著劣于 CineSync baseline
- [ ] 至少一个消融表明显式分支有正向作用
- [ ] 至少一个结果表明 auditory ROI 或 motion guidance 有实际收益

## G. Deliverables required from Claude
Claude 最终必须给出：
1. 改动文件清单
2. 每个文件改了什么
3. 新增 config 示例
4. 训练命令
5. 验证命令
6. 已完成 / 未完成项列表
7. 当前已知风险