# 发给 Claude 的参考材料清单

**用途**: 当你需要在新会话中与 Claude 讨论 CineBrain-SF v1 的创新点、后续方向、或论文写作时，可以将以下材料一并提供给 Claude 作为上下文。

---

## 必发材料 (核心上下文)

### 1. 方法论文档 (刚整理的)
```
docs/METHOD_PAPER_DRAFT.md
```
- 论文风格的完整方法描述
- 包含架构、Loss、训练策略、实验结果、当前挑战和未来方向
- **建议: 这是最重要的文件，必须首先发送**

### 2. 项目状态文件
```
HANDOFF.md
```
- 当前工作状态、已有实验结果、未解决问题
- 包含 Gating-Quality Pareto Tradeoff 的最新诊断
- 包含所有 checkpoint 路径和评估数据

---

## 强烈建议发送 (设计决策链)

### 3. Fast Branch 重设计文档
```
docs/plans/2026-04-02-fast-branch-redesign.md
```
- 记录了 Fast Branch 三次迭代的完整推理过程
- 包含文献调研、失败分析、P0/P1 设计思路
- **这份文档对理解"为什么这么设计"至关重要**

### 4. 优化路线图 (文献调研 + 技术方案)
```
docs/plans/2026-04-06-optimization-roadmap.md
```
- Phase 0-3 的完整优化规划
- 包含 6 篇竞品论文的技术分析
- 每个优化方向的预期收益和实现难度评估
- **讨论创新点时最有用的参考**

### 5. 解决方案分析 (Gating 问题)
```
SOLUTION_ANALYSIS.md
```
- Router Loss 帕累托困境的深度分析
- 6 种方案 (Focal Loss, Weight Annealing, Gradient Reversal 等) 的对比
- **如果讨论当前 blocker 问题，必须提供**

---

## 可选材料 (按需发送)

### 6. 方法规范 (形式化定义)
```
docs/02_METHOD_SPEC.md
```
- 输入/输出/Loss 的形式化数学定义
- 4 模块架构的原始设计规范

### 7. Stage 2 Fusion 设计
```
docs/plans/2026-04-03-stage2-fusion-design.md
```
- GatedFusion 梯度穿透方案的详细推导
- 冻结策略的设计决策

### 8. Code Review 报告
```
docs/review/REVIEW_REPORT.md
docs/review/REVIEW_RESPONSE.md
```
- 6 维度代码审查报告
- 发现并修复的 6 个关键 bug
- 了解 v1→v2 改进的来龙去脉

### 9. 消融实验设计
```
hypothesis_tree.md
ablation_matrix.md
experiment_results.md
```
- 假设树: H1 (Loss Weighting), H2 (Parent Instability), H3 (Runtime/Config)
- 4 组最小消融实验矩阵
- 实验结果汇总

### 10. 评估协议
```
eval_protocol.md
```
- 标准化评估流程 (mini50/mini200)
- pass/fail 判据

### 11. 项目简介 (高层次)
```
docs/01_PROJECT_BRIEF.md
```
- 项目目标、核心命题、与 CineSync 的差异
- 第一版必须实现的内容清单

### 12. 架构图
```
docs/figure/CineBrain-SF-v1-overview-v2.png
docs/figure/CineBrain-SF-v1-fast-branch-detail.png
docs/figure/CineBrain-SF-v1-fusion-guidance-detail.png
docs/figure/CineBrain-SF-v1-training-stages.png
```
- 系统架构总览、Fast Branch 细节、Fusion+Guidance 细节、训练阶段流程
- Claude 支持看图 (multimodal)，可以直接发送 PNG

### 13. 参考代码 (竞品)
```
_ref/eeg2video/     — EEG2Video (NeurIPS'24) 参考实现
_ref/neuroclips/    — NeuroClips (NeurIPS'24 Oral) 参考实现
```
- 如果讨论具体技术借鉴可以发

### 14. 离线目标提取规范
```
docs/离线 Supervision Target 提取规范文档.md
```
- 详细的 supervision target 提取流程
- keyframe、scene embedding、structure latent、motion/dynamics 提取方法

---

## GitHub 仓库
```
https://github.com/HandsomeDrift/SF-v1
```
- 完整代码，Claude 可以通过 WebFetch 直接查看
- 但注意: 仓库代码可能不是最新的 (ts3 上的实验代码可能更新)

---

## 建议发送策略

### 场景 1: 讨论创新点和论文方向
优先级: ① METHOD_PAPER_DRAFT.md → ② optimization-roadmap → ③ HANDOFF.md → ④ 架构图

### 场景 2: 解决当前 Gating 问题
优先级: ① METHOD_PAPER_DRAFT.md → ② SOLUTION_ANALYSIS.md → ③ HANDOFF.md → ④ hypothesis_tree.md

### 场景 3: 全面上下文恢复 (新 Claude 完全不了解项目)
优先级: ① METHOD_PAPER_DRAFT.md → ② HANDOFF.md → ③ fast-branch-redesign → ④ optimization-roadmap → ⑤ 架构图

### 场景 4: 论文写作
优先级: ① METHOD_PAPER_DRAFT.md → ② 架构图 → ③ 01_PROJECT_BRIEF → ④ REVIEW_REPORT
