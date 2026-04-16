# 实验结果汇总

## 实验目标
验证 Stage 3 gating 退化的根因：Parent instability? Loss weights? Router/Alpha 监督?

---

## 实验设计

### 实验 1: Parent 消融
- 固定配置：lambda_sf=0.01, lambda_router=0.1, flow_codebook_k=64
- 只改变 parent checkpoint

### 实验 2: Weight 消融
- 固定 parent=S2 iter50
- 只改变 loss weights

### 实验 3: Alpha 监督消融
- 固定 parent=S2 iter50, 新权重
- 关闭 lambda_alpha_mot, lambda_alpha_nonbrain, lambda_alpha_motion_margin

---

## 实验结果

### 完整结果表

| 实验 | 配置 | iter50 gating | iter50 结果 | iter200 gating | iter200 结果 |
|------|------|---------------|--------------|----------------|---------------|
| **Exp 1A** | parent=iter50, 新权重 | 0.147 | **4/4** ✓ | -0.164 | 3/4 ✗ |
| **Exp 1B** | parent=iter100, 新权重 | 0.094 | **4/4** ✓ | -0.147 | 3/4 ✗ |
| **Exp 2A** | parent=iter50, 原权重 | 0.0 | 3/4 ✗ | -0.134 | 3/4 ✗ |
| **Exp 3B** | parent=iter50, 无alpha | 0.058 | **4/4** ✓ | -0.079 (iter150) | 3/4 ✗ |

### 关键数据

#### Flow Trajectory (所有实验一致稳定)
- Exp 1A iter50: 0.359
- Exp 1A iter200: 0.359
- Exp 1B iter50: 0.359
- Exp 1B iter200: 0.360
- Exp 2A iter50: 0.360
- Exp 2A iter200: 0.359
- Exp 3B iter50: 0.360
- Exp 3B iter150: 0.359

**结论**: Flow trajectory 完全稳定，与配置无关。

---

## 结论分析

### 1. Parent Instability (实验 1)
- **结论**: 不是主因
- 两种 parent 在 iter50 都正常，iter200 都退化
- 差异不大：iter50 parent (-0.164) vs iter100 parent (-0.147)

### 2. Loss Weights (实验 2)
- **结论**: 有效但不是根本解决方案
- 原权重 (0.003/0.5) 在 iter50 就失败 (gating=0.0)
- 新权重 (0.01/0.1) 延迟到 iter200 才失败
- 改进显著但不能完全解决问题

### 3. Alpha Losses (实验 3)
- **结论**: 不是唯一关键因素
- 关闭 alpha losses 后 iter50 仍然通过 (gating=0.058)
- 说明有其他因素帮助短期恢复

### 4. 根本问题
- **长程训练本身导致退化**，不论配置如何
- 即使最佳配置 (新权重 + 健康 parent)，iter200 仍退化
- 需要更根本的架构/优化改进

---

## 下一步建议

1. **检查梯度**：验证训练过程中 router/diffusion 的梯度贡献
2. **更短迭代测试**：确认是否 iter100 是临界点
3. **架构改进**：可能需要修改 gate/router 本身，而不是只调权重
4. **学习率调整**：尝试不同的 learning rate schedule

---

## 产物位置

### Checkpoints
- Exp 1A: `ckpts_5b/exp1a_parent50-04-12-06-54/`
- Exp 1B: `ckpts_5b/exp1b_parent100-04-12-08-28/`
- Exp 2A: `ckpts_5b/exp2a_orig_weights-04-12-14-03/`
- Exp 3B: `ckpts_5b/exp3b_no_alpha-04-12-15-47/`

### Evaluation Results
- `eval_results/exp1a_parent50_iter50_mini50.json`
- `eval_results/exp1a_parent50_iter200_mini50.json`
- `eval_results/exp1b_parent50_iter50_mini50.json`
- `eval_results/exp1b_parent100_iter200_mini50.json`
- `eval_results/exp2a_iter50.json`
- `eval_results/exp2a_iter200.json`
- `eval_results/exp3b_iter50.json`
- `eval_results/exp3b_iter150.json`

### Training Logs
- `logs/exp1a_parent50_gpu1.log`
- `logs/exp1b_parent100_gpu1.log`
- `logs/exp2a_orig_weights_gpu1.log`
- `logs/exp3b_no_alpha_gpu1.log`