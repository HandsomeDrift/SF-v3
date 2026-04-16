# 最小消融实验矩阵 (Ablation Matrix)

## 实验设计原则
- 每组只改变**一个因素**
- 保持其他所有条件完全一致
- 相同训练长度、相同评估集、相同记录指标

---

## Group 1: 方案一 (lambda 调整)

**目的**: 验证"只改权重是否能解决问题"

| 实验名 | 改什么 | 训练长度 | 评估集 | 验证指标 |
|--------|--------|----------|--------|----------|
| exp1a_baseline | 原 recovery-short 配置 (lambda_sf=0.003, lambda_router=0.5) | 100 iter | mini50 | gating_spearman |
| exp1b_weight_only | 仅改权重: lambda_sf=0.01, lambda_router=0.1 | 100 iter | mini50 | gating_spearman |

**预期**: 如果 H1 成立, exp1b 应该比 exp1a 稳定

---

## Group 2: Parent 影响

**目的**: 验证 H2 (parent instability 是主因)

| 实验名 | 改什么 | 训练长度 | 评估集 | 验证指标 |
|--------|--------|----------|--------|----------|
| exp2a_parent50 | 使用 S2 recovery iter50 作为 parent | 100 iter | mini50 | gating_spearman |
| exp2b_parent100 | 使用 S2 recovery iter100 作为 parent | 100 iter | mini50 | gating_spearman |

**预期**: 如果 H2 成立, exp2b 应该出现退化, exp2a 应该稳定

---

## Group 3: Router Penalties 影响

**目的**: 验证新增的 alpha losses 是否关键

| 实验名 | 改什么 | 训练长度 | 评估集 | 验证指标 |
|--------|--------|----------|--------|----------|
| exp3a_with_alpha | 保留 lambda_alpha_mot=0.1, lambda_alpha_nonbrain=0.15 | 100 iter | mini50 | gating_spearman |
| exp3b_no_alpha | 关闭所有 lambda_alpha_* losses | 100 iter | mini50 | gating_spearman |

**预期**: 如果 router penalties 关键, exp3b 应该退化

---

## Group 4: Runtime Config 影响

**目的**: 验证 flow_codebook_k 和 sparse_attn_drop 的影响

| 实验名 | 改什么 | 训练长度 | 评估集 | 验证指标 |
|--------|--------|----------|--------|----------|
| exp4a_default | 使用默认 model config (flow_codebook_k=0) | 100 iter | mini50 | gating_spearman |
| exp4b_override | 使用 override (flow_codebook_k=64) | 100 iter | mini50 | gating_spearman |

---

## 执行顺序建议

1. **首先执行 exp2 (Parent 消融)** - 因为 H2 最高可能性
2. **然后执行 exp1 (Weight 消融)** - 验证方案一是否真的有效
3. **最后执行 exp3, exp4** - 如果前面的结果不确定

---

## 验证标准

每组实验必须报告:
- `gating_alpha_mot_dyn_spearman`
- `gating_checks_passed` (4/4)
- `alpha_mot_mean`, `alpha_nonbrain_mean`
- 训练日志中的 loss 曲线

**判定**:
- spearman > 0.1 且 checks=4/4: **通过**
- spearman ≤ 0.1 或 checks < 4: **失败**