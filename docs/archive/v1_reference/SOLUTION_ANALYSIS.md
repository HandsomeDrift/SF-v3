# Router Loss 问题分析与解决方案

**日期**: 2026-04-15  
**问题**: lambda_sf_router 调参困境 — 0.8 时 Gating 准确但 FVD 崩塌，0.02 时画质恢复但 Gating 反向

---

## 一、问题本质

### 1.1 当前困境

| lambda_sf_router | Gating (Spearman) | FVD | 结论 |
|------------------|-------------------|-----|------|
| 0.8 (Global 2500) | 0.1054 ✅ | 2785 ❌ | Router 主导，DiT "弃画从教" |
| 0.02 (Resume v3) | -0.14 ❌ | 正常 ✅ | 约束力不足，EEG 分支被抑制 |

### 1.2 根本矛盾

这是一个**帕累托前沿博弈问题**：

```
目标1: min L_diff (画质)  ←→  目标2: min L_router (分类准确率)
```

当两个目标**本质上冲突**时（如静态场景但 fMRI 信号强），模型必须在两者之间做权衡。

**为什么简单调 lambda 无法根本解决？**

1. **梯度量级差异巨大**: Router BCE loss 的梯度天然比 diffusion loss 强得多（260倍差异）
2. **目标函数不兼容**: DiT 的最优策略是生成符合视频先验的像素，Router 的最优策略是准确分类
3. **扩散模型的"脑信号偏好"**: fMRI 更稳定、更容易拟合，模型会下意识抑制不稳定的 EEG 分支

---

## 二、解决方案：Focal Loss + 权重退火

### 2.1 核心思想

**不是调整权重大小，而是改变损失函数的性质**：

1. **Focal Loss**: 只对难分类样本施加强梯度，对已分对的样本降低权重
2. **Soft Focal Loss**: 更激进，对置信度足够的样本完全不计入损失
3. **权重退火**: 前期强制激活 Gate，后期恢复画质

### 2.2 实现细节

#### Focal Loss

```python
def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    pt = torch.where(target == 1, pred, 1 - pred)  # 预测正确的概率
    focal_weight = (1 - pt) ** gamma  # 难样本权重高
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)  # 类别平衡
    return (alpha_t * focal_weight * bce).mean()
```

**优势**: 
- 对于 `target=1, pred=0.9` 的样本，`focal_weight = 0.1^2 = 0.01`，梯度很小
- 对于 `target=1, pred=0.3` 的样本，`focal_weight = 0.7^2 = 0.49`，梯度正常
- 避免模型为了把 alpha 从 0.9 推到 1.0 而破坏画质

#### Soft Focal Loss（推荐）

```python
def soft_focal_loss(pred, target, gamma=2.0, margin=0.2):
    pt = torch.where(target == 1, pred, 1 - pred)
    mask = (pt < (1 - margin)).float()  # 只对 pt < 0.8 的样本计算损失
    bce = F.binary_cross_entropy(pred, target, reduction='none')
    focal_weight = (1 - pt) ** gamma
    return (mask * focal_weight * bce).sum() / (mask.sum() + 1e-8)
```

**优势**:
- 对于 `target=1, pred>0.8` 的样本，直接 mask 掉，零梯度
- 更激进地保护画质，只关注真正需要纠正的样本

#### 权重退火

```python
def get_router_lambda(self):
    if self.global_step < self.router_warmup_iters:
        return self.router_lambda_start  # 前 500 步保持高权重
    else:
        progress = (self.global_step - warmup) / (2000 - warmup)
        # cosine 衰减到 end
        return end + 0.5 * (start - end) * (1 + cos(pi * progress))
```

**策略**:
- 前 500 步: `lambda=0.3`，强制激活 Gate
- 后 1500 步: cosine 衰减到 `lambda=0.05`，恢复画质

---

## 三、实验配置

### 3.1 激进方案（从 Stage 2 开始）

**配置文件**: `configs/sf_v1/sf_v1_stage3_joint_focal.yaml`

```yaml
router_loss_type: soft_focal
router_focal_gamma: 2.0
router_lambda_schedule: cosine_warmup
router_lambda_start: 0.3      # 前 500 步强制激活
router_lambda_end: 0.05       # 后期恢复画质
router_warmup_iters: 500
```

**适用场景**: 从 Stage 2 checkpoint 开始，需要强力激活 Gate

### 3.2 保守方案（从 Global 1000 恢复）

**配置文件**: `configs/sf_v1/sf_v1_stage3_joint_focal_conservative.yaml`

```yaml
router_loss_type: soft_focal
router_lambda_schedule: cosine_warmup
router_lambda_start: 0.15     # 更温和的初始权重
router_lambda_end: 0.03       # 更温和的最终权重
router_warmup_iters: 150      # 更短的退火周期
train_iters: 300              # mini500 快速验证
load: ckpts_5b/sf_v1_stage3_full_recovery_resume_v3-04-14-19-18/1000
```

**适用场景**: 从画质良好的 checkpoint 恢复，只需微调 Gating

---

## 四、预期效果

### 4.1 Focal Loss 的作用

- **减少梯度冲突**: 只对难样本施加强梯度，避免"为了 0.9→1.0 而破坏画质"
- **自适应调整**: 随着训练进行，易样本自动降权，模型聚焦难样本

### 4.2 权重退火的作用

- **前期**: 高权重强制激活 Gate，打破 Stage 2 的死区状态
- **后期**: 低权重恢复画质，让 DiT 专注于生成

### 4.3 成功标准

| 指标 | 目标 |
|------|------|
| Gating Spearman | > 0.05 (正相关) |
| FVD | < 700 (接近 SOTA 618) |
| Pearson r | > 0.30 (运动建模) |
| L_temp_delta | < 0.05 (时序稳定) |

---

## 五、下一步行动

### 5.1 立即执行（保守方案）

```bash
# 1. 检查 GPU 占用
ssh ts3 "ssh gpu2 'nvidia-smi'"

# 2. 启动 mini500 快速验证（300 iter）
ssh ts3 "ssh gpu2 'cd /public/home/maoyaoxin/zhangt/xxt/SF-v1/CineBrain && \
  CUDA_HOME=/usr/local/cuda-12.4 \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  nohup /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python \
  train_video_fmri.py --base configs/sf_v1/cinebrain_sf_v1_model.yaml \
  configs/sf_v1/sf_v1_stage3_joint_focal_conservative.yaml \
  --seed 42 > logs/focal_conservative_test.log 2>&1 &'"

# 3. 监控日志
ssh ts3 "tail -f /public/home/maoyaoxin/zhangt/xxt/SF-v1/CineBrain/logs/focal_conservative_test.log"
```

### 5.2 评估检查点

```bash
# 每 100 iter 评估一次
ssh ts3 "ssh gpu2 'cd /public/home/maoyaoxin/zhangt/xxt/SF-v1/CineBrain && \
  CUDA_VISIBLE_DEVICES=4 \
  /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python \
  tools/evaluate_p1.py \
  --ckpt ckpts_5b/sf_v1_stage3_joint_focal_conservative-XX-XX-XX-XX/100 \
  --data_json /public/home/maoyaoxin/zhangt/xxt/datasets/sub-0005_train_va_mini50.json \
  --output eval_results/focal_conservative_iter100.json'"
```

### 5.3 如果成功，扩展到全量训练

修改配置：
- `train_data`: mini500 → 完整数据集
- `train_iters`: 300 → 2000
- `load`: 从最佳 mini500 checkpoint 继续

---

## 六、备选方案（如果 Focal Loss 仍不够）

### 6.1 分离优化器

为 `gate_net` 单独创建优化器，完全解耦 Router 和 DiT 的训练。

**优点**: 彻底解耦  
**缺点**: 需要修改训练循环，工程量大

### 6.2 Gradient Reversal Layer

借鉴域适应，让 Router 学习分类的同时，反向梯度阻止 DiT 破坏画质。

**优点**: 理论优雅  
**缺点**: 实现复杂，调试困难

---

## 七、总结

**核心洞察**: 这不是权重调参问题，而是**损失函数设计问题**。

- **旧方案**: BCE loss 对所有样本一视同仁 → 模型为了降低 loss 牺牲画质
- **新方案**: Focal Loss 只关注难样本 + 权重退火 → 前期激活 Gate，后期恢复画质

**预期**: 在 0.15-0.03 的温和权重下，通过 Soft Focal Loss 的自适应特性，找到 Gating 和画质的平衡点。
