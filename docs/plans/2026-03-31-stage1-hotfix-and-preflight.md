# Stage I 热修复 + 训练前置验证 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复 Stage I 训练中 sf_targets 未加载、supervision target 格式不匹配、auditory ROI 未分离等问题，并建立训练前置验证体系（preflight check），确保类似问题不再发生。

**Architecture:** 分两大阶段 —— Phase A 修复所有已知 bug（6 个 Task），Phase B 建立 preflight 验证体系（2 个 Task）。修复完成后先跑 preflight 验证，通过后再启动 Stage I 重训。

**Tech Stack:** PyTorch, DeepSpeed, SAT framework, CogVideoX-5B, SigLIP, RAFT optical flow

---

## 问题清单

| # | 严重性 | 问题 | 影响 |
|---|--------|------|------|
| P1 | **致命** | `data_video.py` v1_per_clip 格式的 sf_targets 不加载 `_sf_preloaded` | 全部 branch head supervision = 0 |
| P2 | **致命** | Fast branch targets 格式错误：v1 targets 全是 (1152) SigLIP embed，但 z_dyn=(B,), z_mot=(B,1920), z_tc=(B,) | shape mismatch 或静默回归错误目标 |
| P3 | **致命** | `gt_structure_embed` 不存在于 sf_targets → StructureHead 无 supervision | L_str = 0 |
| P4 | **严重** | AlignmentLoss 未接收 video/text embed → 只有 L_fe | 4/5 对比学习 loss 缺失 |
| P5 | **重要** | fMRI 18946 维混合 visual+auditory 但未分离 → AudiovisualContextAdapter 无法启用 | 无法利用听觉皮层信息 |
| P6 | **重要** | 无训练前置验证 → 花 14 小时才发现 loss 全错 | 浪费 GPU 时间 |

## 修复策略

**对 P2/P3 的关键决策：统一为 SigLIP 1152-dim targets，调整 head 输出维度匹配**

理由：
- v2 extraction（VAE latent + RAFT flow）之前提取过但 sharding 失败，重新提取耗时且引入额外依赖
- Stage I 的核心目标是验证 slow-fast 分支架构是否有效，SigLIP embedding 作为 supervision target 已经足够有意义（keyframe=视觉语义, text=文本语义, dynamics=运动强度语义, motion=运动模式语义, tc=时间连贯性语义）
- 先用 SigLIP targets 跑通，后续再用 RAFT/VAE targets 做消融实验

具体变更：
- `DynamicsHead`: out_dim=1 → out_dim=1152（回归 SigLIP dynamics embedding）
- `MotionHead`: out_dim=1920 → out_dim=1152（回归 SigLIP motion embedding）
- `TemporalCoherenceHead`: out_dim=1 → out_dim=1152（回归 SigLIP tc embedding）
- `StructureHead`: 暂时禁用（无 gt_structure_embed target）
- `MultiGuidanceAdapter.mot_input_dim`: 1922 → 3456（1152×3）

---

## Phase A: Bug 修复

### Task 1: 修复 sf_targets v1_per_clip 加载 [P1]

**Files:**
- Modify: `CineBrain/data_video.py` (BrainDataset.__init__ 和 get_item_func)

**问题根因:** `_sf_preloaded` 只在 `v2_sharded` 格式时填充，v1_per_clip 格式被检测到但从未加载数据。

**Step 1: 在 BrainDataset.__init__ 中添加 v1_per_clip 预加载**

在 `__init__` 方法中，现有 v2_sharded 预加载代码块之后，添加 v1_per_clip 加载逻辑：

```python
        # Preload v1 per-clip sf_targets (directory-based: sf_targets/{clip_id}/{target}.npy)
        if getattr(self, '_sf_cache_format', None) == "v1_per_clip" and self.sf_targets_dir:
            import glob as _glob
            clip_dirs = sorted([d for d in os.listdir(self.sf_targets_dir)
                                if os.path.isdir(os.path.join(self.sf_targets_dir, d))])
            for clip_dir in clip_dirs:
                clip_id = int(clip_dir)
                clip_path = os.path.join(self.sf_targets_dir, clip_dir)
                entry = {}
                for fname in os.listdir(clip_path):
                    fpath = os.path.join(clip_path, fname)
                    key = os.path.splitext(fname)[0]  # e.g. "gt_keyframe_embed"
                    if fname.endswith(".npy"):
                        entry[key] = torch.from_numpy(np.load(fpath))
                    elif fname.endswith(".pt"):
                        entry[key] = torch.load(fpath, map_location="cpu", weights_only=False)
                self._sf_preloaded[clip_id] = entry
            print(f"[BrainDataset] Preloaded {len(self._sf_preloaded)} v1 per-clip targets")
```

**Step 2: 修复 get_item_func 中的 key 映射**

v1 targets 的 key 已经是最终名称（gt_keyframe_embed 等），不需要 rename。但当前代码用 v2 格式的 key 做 rename，对 v1 数据无效。修改 get_item_func 中的 sf_targets 加载段：

```python
        sf_targets = {}
        if clip_id_int in self._sf_preloaded:
            sf_targets = dict(self._sf_preloaded[clip_id_int])
            # v2 sharded format uses different key names, rename to standard
            if getattr(self, '_sf_cache_format', None) == "v2_sharded":
                _name_map = {
                    "keyframe_img_emb": "gt_keyframe_embed",
                    "scene_text_emb": "gt_text_embed",
                    "structure_latent": "gt_structure_embed",
                    "flow_token": "gt_motion_embed",
                    "flow_mag": "gt_dynamics_embed",
                    "ofs_score": "gt_tc_embed",
                }
                for src, dst in _name_map.items():
                    if src in sf_targets:
                        sf_targets[dst] = sf_targets.pop(src)
        item["sf_targets"] = sf_targets
```

**Step 3: 验证**

```bash
# 在 gpu2 上运行快速验证
python -c "
from data_video import BrainDataset
ds = BrainDataset('/path/to/sub-0005_train_va.json', video_size=[480,720], fps=8, max_num_frames=33, skip_frms_num=0)
item = ds[0]
print('sf_targets keys:', list(item['sf_targets'].keys()))
for k, v in item['sf_targets'].items():
    print(f'  {k}: shape={v.shape}, dtype={v.dtype}')
assert len(item['sf_targets']) > 0, 'sf_targets is empty!'
print('PASS: sf_targets loaded successfully')
"
```

**Step 4: Commit**

---

### Task 2: 调整 Fast Branch heads 为 1152-dim 输出 [P2]

**Files:**
- Modify: `CineBrain/sgm/modules/encoders/fast_branch.py`

**Step 1: 修改 DynamicsHead, MotionHead, TemporalCoherenceHead 输出维度**

将三个 head 统一改为输出 (B, 1152)，与 SigLIP embedding targets 对齐：

```python
class DynamicsHead(nn.Module):
    """Predict dynamics embedding z_dyn from fast features.
    Supervised by SigLIP dynamics embedding (1152-dim).
    """
    def __init__(self, in_dim=2048, out_dim=1152):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_dyn: (B, out_dim) via mean pooling + MLP"""
        return self.proj(x.mean(dim=1))


class MotionHead(nn.Module):
    """Predict motion embedding z_mot from fast features.
    Supervised by SigLIP motion embedding (1152-dim).
    """
    def __init__(self, in_dim=2048, out_dim=1152):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_mot: (B, out_dim) via mean pooling + MLP"""
        return self.proj(x.mean(dim=1))


class TemporalCoherenceHead(nn.Module):
    """Predict temporal coherence embedding z_tc from fast features.
    Supervised by SigLIP temporal coherence embedding (1152-dim).
    """
    def __init__(self, in_dim=2048, out_dim=1152):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x):
        """x: (B, S, D) → z_tc: (B, out_dim) via mean pooling + MLP"""
        return self.proj(x.mean(dim=1))
```

**Step 2: 更新 FastBranch 构造参数**

移除不再需要的 `motion_token_dim` 参数，各 head 统一使用 `head_dim`（默认 1152）：

```python
class FastBranch(nn.Module):
    def __init__(
        self,
        eeg_encoder,
        embed_dim=2048,
        head_dim=1152,
        use_dynamics_head=True,
        use_motion_head=True,
        use_temporal_coherence_head=True,
    ):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.use_dynamics_head = use_dynamics_head
        self.use_motion_head = use_motion_head
        self.use_temporal_coherence_head = use_temporal_coherence_head

        if use_dynamics_head:
            self.dynamics_head = DynamicsHead(embed_dim, out_dim=head_dim)
        if use_motion_head:
            self.motion_head = MotionHead(embed_dim, out_dim=head_dim)
        if use_temporal_coherence_head:
            self.tc_head = TemporalCoherenceHead(embed_dim, out_dim=head_dim)
```

**Step 3: 更新 fast_branch.forward 的 docstring**

把 `"z_dyn": (B,)` 等全部改为 `"z_dyn": (B, 1152)` 等。

**Step 4: Commit**

---

### Task 3: 更新 MultiGuidanceAdapter 和 sf_losses [P2 延续]

**Files:**
- Modify: `CineBrain/sgm/modules/encoders/multi_guidance.py`
- Modify: `CineBrain/sgm/modules/diffusionmodules/sf_losses.py`

**Step 1: 更新 MultiGuidanceAdapter 的 mot_input_dim**

现在 z_dyn, z_mot, z_tc 都是 (B, 1152)，cat 后为 (B, 3456)：

```python
class MultiGuidanceAdapter(nn.Module):
    def __init__(
        self,
        brain_dim=4096,
        head_dim=1152,
        num_spatial=226,
        use_keyframe_guidance=True,
        use_text_guidance=True,
        use_motion_guidance=True,
        use_brain_latent_guidance=True,
        mot_input_dim=3456,  # 1152 * 3 (z_dyn + z_mot + z_tc)
    ):
```

同时简化 `forward` 中 motion guidance 的拼接逻辑（不再需要 squeeze/unsqueeze scalar 处理）：

```python
        if self.use_motion_guidance:
            B = z_b.shape[0]
            device, dtype = z_b.device, z_b.dtype
            z_dyn = fast_out.get("z_dyn", torch.zeros(B, self.mot_input_dim // 3, device=device, dtype=dtype))
            z_mot = fast_out.get("z_mot", torch.zeros(B, self.mot_input_dim // 3, device=device, dtype=dtype))
            z_tc = fast_out.get("z_tc", torch.zeros(B, self.mot_input_dim // 3, device=device, dtype=dtype))
            mot_cat = torch.cat([z_dyn, z_mot, z_tc], dim=-1)  # (B, 3456)
            g_mot = self.mot_proj(mot_cat).unsqueeze(1)
            context = context + alphas["alpha_mot"].unsqueeze(-1) * g_mot
```

**Step 2: 更新 sf_embedder.py 中 mot_input_dim 参数**

```python
            self.guidance_adapter = MultiGuidanceAdapter(
                ...
                mot_input_dim=clip_dim * 3,  # 1152 * 3 = 3456
            )
```

**Step 3: 简化 FastBranchLoss**

移除 `reshape_as` 逻辑，所有 fast branch loss 统一为 (B, 1152) 对 (B, 1152) 的 MSE：

```python
class FastBranchLoss(nn.Module):
    def forward(self, fast_out, targets):
        _ref = next(iter(fast_out.values()))
        losses = {}
        total = _ref.new_tensor(0.0)

        if "z_dyn" in fast_out and "gt_dynamics_embed" in targets:
            losses["L_dyn"] = F.mse_loss(fast_out["z_dyn"], targets["gt_dynamics_embed"])
            total = total + self.lambda_dyn * losses["L_dyn"]
        if "z_mot" in fast_out and "gt_motion_embed" in targets:
            losses["L_mot"] = F.mse_loss(fast_out["z_mot"], targets["gt_motion_embed"])
            total = total + self.lambda_mot * losses["L_mot"]
        if "z_tc" in fast_out and "gt_tc_embed" in targets:
            losses["L_tc"] = F.mse_loss(fast_out["z_tc"], targets["gt_tc_embed"])
            total = total + self.lambda_tc * losses["L_tc"]

        return total, losses
```

**Step 4: 暂时禁用 StructureHead 的 loss（无 target）**

在 `SlowBranchLoss.forward` 中，保留 `L_str` 逻辑不变（`gt_structure_embed` 不在 targets 中时自动跳过）。

在 `sf_embedder.py` 配置中设置 `use_structure_head: False`。

**Step 5: Commit**

---

### Task 4: 修复 AlignmentLoss 接收 video/text embed [P4]

**Files:**
- Modify: `CineBrain/sgm/modules/diffusionmodules/loss.py` (VideoDiffusionLossSF.__call__)

**Step 1: 将 gt_keyframe_embed 和 gt_text_embed 传入 AlignmentLoss**

```python
        # Alignment loss
        if "fmri_cls" in slow_out and "eeg_cls" in fast_out:
            video_embed = targets.get("gt_keyframe_embed", None)
            text_embed = targets.get("gt_text_embed", None)
            l_align, _ = self.align_loss(slow_out, fast_out, video_embed, text_embed)
            sf_total = sf_total + l_align
```

**Step 2: Commit**

---

### Task 5: 实现 fMRI auditory ROI 分离 [P5]

**Files:**
- Modify: `CineBrain/data_video.py` (BrainDataset.get_item_func)
- Modify: `CineBrain/sgm/modules/encoders/sf_embedder.py` (SFBrainEmbedder)
- Modify: `CineBrain/sgm/modules/encoders/slow_branch.py` (SlowBranch)
- Modify: `CineBrain/configs/sf_v1/cinebrain_sf_v1_model.yaml`

**Step 1: 在 data_video.py 中拆分 fMRI 为 visual + auditory**

在 `get_item_func` 中，加载 fmri 后拆分：

```python
        fmri = torch.cat(fmri_data_list, dim=0)  # (5, 18946)
        # Split visual (8405) and auditory (10541) ROIs
        fmri_visual = fmri[:, :8405]      # (5, 8405)
        fmri_auditory = fmri[:, 8405:]    # (5, 10541)

        # ...
        item = {
            "mp4": tensor_frms,
            "fmri": fmri_visual,           # 改为只传 visual ROI
            "fmri_auditory": fmri_auditory, # 新增 auditory ROI
            "eeg": eeg,
            # ... 其他字段不变
        }
```

**Step 2: 在 train_video_fmri.py 的 broad_cast_batch 中广播 fmri_auditory**

类似 eeg 的广播逻辑，添加 fmri_auditory 的广播：

```python
    # Broadcast fmri_auditory tensor
    aud_meta = [batch.get("fmri_auditory") is not None,
                batch["fmri_auditory"].shape if batch.get("fmri_auditory") is not None else None]
    torch.distributed.broadcast_object_list(aud_meta, src=src, group=mpu.get_model_parallel_group())
    if aud_meta[0]:
        if mpu.get_model_parallel_rank() != 0:
            batch["fmri_auditory"] = torch.zeros(aud_meta[1], device="cuda")
        batch["fmri_auditory"] = batch["fmri_auditory"].contiguous()
        torch.distributed.broadcast(batch["fmri_auditory"], src=src, group=mpu.get_model_parallel_group())
```

同时在非 rank-0 的 batch 初始化中添加 `"fmri_auditory": None`。

**Step 3: 在 SFBrainEmbedder 中添加 auditory encoder**

```python
        # Auditory encoder (reuse fMRI encoder architecture with different seq_len)
        if use_auditory:
            self.auditory_encoder = CustomfMRITransformer(
                clip_dim=clip_dim, in_channels=in_channels,
                seq_len=10541,  # auditory ROI voxels
                num_layers=fmri_num_layers // 2,  # lighter: 6 layers vs 12
                num_spatial=num_spatial
            )
```

修改 `forward` 中 slow_branch 的调用：

```python
        if self.use_slow_branch and self.use_fast_branch:
            auditory_fmri = batch.get("fmri_auditory")
            if auditory_fmri is not None:
                auditory_fmri = auditory_fmri.to(self.dtype)
            slow_out = self.slow_branch(fmri, auditory_fmri=auditory_fmri)
```

**Step 4: 在 SlowBranch 中传入 auditory_encoder**

修改 `sf_embedder.py` 中构造 SlowBranch 的代码：

```python
        if use_slow_branch:
            self.slow_branch = SlowBranch(
                fmri_encoder=self.fmri_encoder,
                auditory_encoder=self.auditory_encoder if use_auditory else None,
                embed_dim=embed_dim,
                head_dim=clip_dim,
                use_auditory=use_auditory,
                ...
            )
```

**Step 5: 更新配置文件**

`configs/sf_v1/cinebrain_sf_v1_model.yaml`:
```yaml
  seq_len: 8405          # 改为只有 visual ROI
  use_auditory: True     # 启用听觉分支
  use_structure_head: False  # 暂时禁用（无 target）
```

**Step 6: 更新 diffusion_video_brain.py disable_untrainable_params**

在 sf_trainable_keys 中添加 `"auditory_encoder"`。

**Step 7: Commit**

---

### Task 6: 更新训练配置 + 清理旧 checkpoint

**Files:**
- Modify: `CineBrain/configs/sf_v1/cinebrain_sf_v1_model.yaml`
- Modify: `CineBrain/configs/sf_v1/sf_v1_train_stage1.yaml`

**Step 1: 更新模型配置**

```yaml
conditioner_config:
  params:
    emb_models:
      - params:
          seq_len: 8405          # visual ROI only (was 18946)
          use_auditory: True     # enable auditory branch
          use_structure_head: False  # no target yet
```

**Step 2: 确认训练配置不变** (5000 iters, 4 GPU, bs=1, grad_accum=4)

**Step 3: 考虑是否删除旧 checkpoint 释放空间**

10 个 checkpoint × 30GB = 300GB。保留 iter 5000 作为参考，其他可删。

**Step 4: Commit**

---

## Phase B: 训练前置验证体系 (Preflight Check)

### Task 7: 创建 preflight 验证脚本

**Files:**
- Create: `CineBrain/tools/preflight_check.py`

**目的:** 在正式训练之前用 1-2 个 step 验证所有关键路径均正常工作。

**脚本内容:**

```python
"""
Preflight Check for SF v1 Training
===================================
在正式训练前运行，验证以下 7 项：

1. sf_targets 加载: batch["sf_targets"] 非空
2. shape 匹配: 所有 head 输出 vs target shape 一致
3. loss 分解: 每项子 loss 非零
4. 梯度流动: 所有可训练参数收到非零梯度
5. auditory ROI: fmri_auditory 存在且 shape 正确
6. 前向/后向无报错: 完整 1 step 无异常
7. 显存预估: 打印峰值显存

用法:
    torchrun --nproc_per_node=1 tools/preflight_check.py \
        --base configs/sf_v1/cinebrain_sf_v1_model.yaml \
              configs/sf_v1/sf_v1_train_stage1.yaml

所有检查通过才允许启动正式训练。
"""
import os, sys, argparse, torch, yaml, json
import torch.distributed as dist

def main():
    # 1. 初始化环境
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    results = {}

    # 2. 加载配置
    # ... 加载 yaml 配置, 构建 model, dataset

    # 3. Check 1: sf_targets 非空
    item = dataset[0]
    st = item["sf_targets"]
    results["sf_targets_loaded"] = len(st) > 0
    print(f"[CHECK 1] sf_targets loaded: {len(st)} keys → {'PASS' if results['sf_targets_loaded'] else 'FAIL'}")
    for k, v in st.items():
        print(f"  {k}: shape={v.shape}")

    # 4. Check 2: shape 匹配
    # 模拟一次 forward，检查 head 输出 vs target shape
    # ...

    # 5. Check 3: loss 分解
    # 打印每项子 loss 的值，确保非零
    # ...

    # 6. Check 4: 梯度流动
    # 1 step backward，检查所有可训练参数的 grad 不为 None
    # ...

    # 7. Check 5: auditory ROI
    # 检查 batch 中 fmri_auditory 存在且 shape = (B, 5, 10541)
    # ...

    # 8. Check 6: 前向/后向无异常
    # try/except 包裹完整 forward+backward step
    # ...

    # 9. Check 7: 显存
    print(f"[CHECK 7] Peak GPU memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # 10. 总结
    all_pass = all(results.values())
    print(f"\n{'='*60}")
    print(f"PREFLIGHT {'PASSED ✅' if all_pass else 'FAILED ❌'}")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL ❌'}")
    if not all_pass:
        print("\n⚠️  DO NOT start training until all checks pass!")
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Step 1: 实现完整的 preflight_check.py**

（包含上述 7 项检查的完整实现）

**Step 2: 运行验证**

```bash
cd /public/home/maoyaoxin/xxt/SF-v1/CineBrain
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 tools/preflight_check.py \
    --base configs/sf_v1/cinebrain_sf_v1_model.yaml configs/sf_v1/sf_v1_train_stage1.yaml
```

Expected: 全部 7 项 PASS

**Step 3: Commit**

---

### Task 8: 创建 loss 分解日志（训练时持续监控）

**Files:**
- Modify: `CineBrain/sgm/modules/diffusionmodules/loss.py` (VideoDiffusionLossSF)

**目的:** 训练过程中记录每项子 loss，方便监控各分支是否真正在学习。

**Step 1: 在 VideoDiffusionLossSF.__call__ 中添加子 loss 记录**

将各子 loss 存入 `self._last_loss_breakdown`，使其可被训练循环 log：

```python
        # 在 return 之前:
        self._last_loss_breakdown = {
            "sf/L_align": l_align.item() if 'l_align' in dir() else 0.0,
            "sf/L_slow": l_slow.item() if 'l_slow' in dir() else 0.0,
            "sf/L_fast": l_fast.item() if 'l_fast' in dir() else 0.0,
            "sf/L_guide": l_guide.item() if 'l_guide' in dir() else 0.0,
            "sf/total": sf_total.item(),
        }
```

**Step 2: 在 shared_step 中传递子 loss 到 loss_dict**

在 `diffusion_video_brain.py` 的 `shared_step` 中，从 loss_fn 获取分解信息并加入 loss_dict：

```python
        # After loss computation:
        if hasattr(self.loss_fn, '_last_loss_breakdown'):
            for k, v in self.loss_fn._last_loss_breakdown.items():
                loss_dict[k] = torch.tensor(v, dtype=torch.float32)
```

**Step 3: Commit**

---

## 执行顺序

```
Task 1 (sf_targets loading fix)
  ↓
Task 2 (fast branch head dims)
  ↓
Task 3 (multi_guidance + sf_losses update)
  ↓
Task 4 (alignment loss fix)
  ↓
Task 5 (auditory ROI separation)
  ↓
Task 6 (config update)
  ↓
Task 7 (preflight check script)  ← 运行并验证所有修复
  ↓
Task 8 (loss decomposition logging)
  ↓
🚀 启动 Stage I 重训
```

## 预估时间

- Phase A (Task 1-6): ~2 小时编码 + 调试
- Phase B (Task 7-8): ~1 小时
- Preflight 验证: ~5 分钟
- Stage I 重训: ~14 小时（与首次相同）
