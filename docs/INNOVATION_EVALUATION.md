# CineBrain-SF v1 — 当前方案客观评估

**Date**: 2026-04-16
**Purpose**: 客观记录当前方案的优劣势，指导后续创新方向

---

## 创新性评估

### 有价值的洞察
1. fMRI+EEG 多模态互补建模（领域内少见的双模态方案）
2. EEG 不能做光流回归的实验性证明（能力边界分析）
3. P0→P1 蒸馏到时序动态的演化路径（完整逻辑链）

### 创新性不足
大部分技术组件是已有方法的组合：
- Cross-attention temporal decoder → 标准 Transformer decoder
- Sparse causal attention → Mind-Animator (ICLR'25)
- Flow codebook → DecoFuse (arXiv'25)
- L_Struct → DynaMind (arXiv'25)
- MoCo queue → MoCo v2 (2020)
- DANA → EEG2Video (NeurIPS'24)

**核心问题**: 审稿人会问 "你们的核心技术创新点是什么？"

## 技术贡献质量

### 扎实的部分
- 系统工程质量高
- 失败分析透彻
- 多阶段训练梯度分析严谨

### 薄弱的部分
1. **Gating 机制**: 核心 claim 是"显式角色分配"，但 gating Spearman 最好只有 0.105，且存在画质-gating 帕累托矛盾
2. **缺乏关键消融**: Slow only vs Slow+P0 vs Slow+P0+P1 未做
3. **Backbone 优势混淆**: CogVideoX-5B 远强于竞品的 SD 1.5 / AnimateDiff

## 实验验证评估

### 做得好的
- FVD ↓31%、EPE ↓20%
- 跨被试泛化验证
- 14 项指标全面评估

### 欠缺的
- Slow only vs Slow+Fast 消融（致命缺失）
- 按动态程度分子集的指标对比
- Gating 行为的定量可视化
- 生成视频定性展示

## 论文竞争力判断

| 维度 | 评分 | 评语 |
|------|------|------|
| 工程质量 | A | 模块化设计、文档完备 |
| 实验结果 | A- | FVD/EPE 强，消融不完整 |
| 问题动机 | B+ | 有说服力但不算新颖 |
| 技术创新 | C+ | 主要是已有方法组合 |
| 故事完整性 | B- | Gating 问题使核心 claim 打折 |
| 论文竞争力 | B | 二区期刊有把握，顶会需补强 |

**一句话**: 工程扎实、结果不错的系统工作，但学术创新深度需要在后续迭代中补强。
