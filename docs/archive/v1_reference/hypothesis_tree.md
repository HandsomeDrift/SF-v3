# 假设树 (Hypothesis Tree)

## 核心现象
- Stage 2/3 在 iter50 时 gating 正常 (spearman ~0.12-0.20)
- Stage 2/3 在 iter100+ 时 gating 退化 (spearman → 0 或负值)
- 即使修改 lambda_sf/lambda_router，问题依然存在

---

## H1: Loss Weighting 问题

**假设**: `lambda_sf` 太小 / `lambda_router` 通路过强导致 router 主导训练并压制 gating 语义

### 支持证据
- 原始配置 `lambda_sf=0.003`, `lambda_router=0.5` 差距 167x
- 修改后 `lambda_sf=0.01`, `lambda_router=0.1` 差距 10x

### 反证
- 旧配置 `sf_v1_stage3_joint.yaml` 根本没有 `lambda_router` (无 router-specific losses)
- 真正跑通的 recovery-short 日志显示 `raw_router_total` 只有 0.03-0.10, 而 `raw_sf_total` 是 100-150
- 缩放后 `sf/router_total ≈ 0.016-0.05`, 不显示 router 主导

### 缺失证据
- **缺少梯度范数记录**: 无法确认 router 梯度是否真的在共享参数上更大
- **缺少梯度方向记录**: 无法确认 router 梯度是否与主目标冲突

### 结论状态: **支持但未证实**

---

## H2: Parent Instability 继承

**假设**: 进入 Stage 3 前, parent checkpoint (Stage 2) 在更长训练时已经不稳定

### 支持证据
- Stage 2 iter50: gating spearman = 0.124 (正常)
- Stage 2 iter100: gating spearman = **0.0** (已退化!)
- Stage 3 从 iter50 parent 启动, 在 iter100 仍出现退化

### 反证
- Stage 3 iter50 评估是正常的 (0.204), 说明从健康的 parent 启动可以恢复

### 缺失证据
- 需要对比: 从 iter50 vs iter100 parent 启动 Stage 3 的消融

### 结论状态: **高度可疑, 需要消融验证**

---

## H3: Runtime/Config 组合效应

**假设**: short-run 改善来自 parent 选择 + router penalties + runtime override + eval config 的组合, 不是单一 lambda 调整

### 支持证据
- 有效 short-run 同时改变了:
  1. Parent: 使用 S2 recovery iter50
  2. Router penalties: 新增 `lambda_alpha_mot/nonbrain/motion_margin`
  3. Runtime override: `flow_codebook_k=64`, `sparse_attn_drop=0.3`
  4. Eval config: 显式使用 `model_phase1_eval.yaml`

### 反证
- 需要单独消融每个因素

### 缺失证据
- 需要分离每个因素的独立贡献

### 结论状态: **需要消融分离**

---

## 判定: 当前最高可能性

基于现有证据, **H2 (Parent Instability)** 是最可能的核心原因:
1. S2 iter100 自己在 mini50 上已经 gating 失败
2. S3 从 iter50 启动能正常, 从 iter100 启动会退化
3. 这解释了为什么"方案一只改了权重"无法根本解决问题

**建议优先级**:
1. 首先验证 H2: 做 parent 消融
2. 其次验证 H1: 加梯度记录
3. 最后验证 H3: 分离 config 组合因素