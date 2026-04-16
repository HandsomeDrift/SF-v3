# 统一评估协议 (Unified Evaluation Protocol)

## 目标
确保所有评估结论基于相同的配置和流程，避免因评估路径不一致导致的误判。

---

## 有效评估配置

### 必须使用的 model config
```bash
configs/sf_v1/_pipeline_tmp/model_phase1_eval.yaml
```

**原因**: 默认 `cinebrain_sf_v1_model.yaml` 中 `flow_codebook_k=0` 会导致 shape mismatch

---

## 标准评估命令

### 1. Mini50 评估 (快速迭代)
```bash
cd /public/home/maoyaoxin/xxt/SF-v1/CineBrain

# 使用 1 GPU
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluate_p1.py \
  --ckpt <checkpoint_path>/mp_rank_00_model_states.pt \
  --data-json sub-0005_test_va.json \
  --model-config configs/sf_v1/_pipeline_tmp/model_phase1_eval.yaml \
  --max-samples 50 \
  --output eval_results/<experiment_name>_iter<iter>_mini50.json
```

### 2. Mini200 评估 (统计验证)
```bash
CUDA_VISIBLE_DEVICES=0 \
python tools/evaluate_p1.py \
  --ckpt <checkpoint_path>/mp_rank_00_model_states.pt \
  --data-json sub-0005_test_va.json \
  --model-config configs/sf_v1/_pipeline_tmp/model_phase1_eval.yaml \
  --max-samples 200 \
  --output eval_results/<experiment_name>_iter<iter>_mini200.json
```

---

## 评估必须记录的信息

每次评估必须记录以下元信息到 JSON:

```json
{
  "checkpoint": "ckpts_5b/...",
  "iteration": 100,
  "eval_config": "configs/sf_v1/_pipeline_tmp/model_phase1_eval.yaml",
  "num_samples": 50,
  "timestamp": "2026-04-12-...",
  "git_commit": "..."
}
```

---

## 评估结果判定标准

### 通过条件
- `gating_checks_passed >= 4`
- `gating_alpha_mot_dyn_spearman > 0.1`
- `temporal_delta_pearson > 0.3`
- `flow_traj_pearson > 0.3`

### 失败条件
- 任一条件不满足

---

## 禁止事项

1. **不要**混用不同 eval config 的结果
2. **不要**把启动失败的日志当作模型结论
3. **不要**把不同 sample 数量的结果直接对比
4. **不要**跳过 model_phase1_eval.yaml 直接用默认 model

---

## 验证检查清单

在报告评估结果前，确认:
- [ ] 使用了 `model_phase1_eval.yaml`
- [ ] 记录了 checkpoint 路径和 iteration
- [ ] 记录了 sample 数量
- [ ] 报告了所有 4 个检查项