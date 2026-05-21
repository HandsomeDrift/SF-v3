# Stage 2 Fusion Training Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable Stage 2 fusion training where L_diff gradient flows through frozen DiT to train GatedFusion + MultiGuidanceAdapter.

**Architecture:** Three code changes (loss.py enable L_diff for fusion stage, diffusion_video_brain.py add unfreeze_fusion logic, new training config yaml) + validation pipeline (preflight → overfit → mini train → full train).

**Tech Stack:** PyTorch, DeepSpeed, SAT framework, CogVideoX-5B DiT

---

## Task 1: Enable L_diff in "fusion" training stage

**Files:**
- Modify: `/home/drift/ts3/SF-v1/CineBrain/sgm/modules/diffusionmodules/loss.py:282-283`

**Step 1: Make the change**

At line 282-283, change:
```python
        # Diffusion loss (stage 3 only)
        if self.training_stage == "joint":
```
to:
```python
        # Diffusion loss (stage 2 fusion + stage 3 joint)
        if self.training_stage in ("fusion", "joint"):
```

This single-line change makes L_diff available in the "fusion" training stage. The rest of the L_diff computation block (lines 284-341) remains unchanged.

**Step 2: Verify no other references need updating**

Search for `training_stage` in loss.py to confirm the comment at line 360 (`# For branch_pretrain / fusion stages: return SF loss only`) is now stale — update it:
```python
        # For branch_pretrain stage: return SF loss only (no diffusion)
        return sf_total.to(input.dtype)
```

Note: The `return` at line 361 is now only reached in `branch_pretrain` stage since both `fusion` and `joint` return inside the L_diff block at line 341.

---

## Task 2: Add unfreeze_fusion logic to diffusion_video_brain.py

**Files:**
- Modify: `/home/drift/ts3/SF-v1/CineBrain/diffusion_video_brain.py:157` (after the `_freeze_fast` block)

**Step 1: Add unfreeze_fusion logic**

After line 157 (`p.requires_grad_(False)` in the `_freeze_fast` block), before line 159 (`# Recount trainable`), insert:

```python
            # Stage 2 Fusion: unfreeze fusion modules after global freeze
            _unfreeze_fusion = model_config.get('unfreeze_fusion', False)
            if _unfreeze_fusion:
                fusion_unfrozen = 0
                for n, p in self.named_parameters():
                    if "gated_fusion" in n or "guidance_adapter" in n:
                        p.requires_grad_(True)
                        fusion_unfrozen += p.numel()
                print_rank0(f"[Stage 2] Unfroze fusion modules: {fusion_unfrozen:,} params")
```

**Step 2: Apply the same change to SATVideoDiffusionEngineBrain_fix**

The same class is duplicated starting at line 407. Apply the identical change after line 537 (the `_freeze_fast` block in the `_fix` class).

---

## Task 3: Create Stage 2 training config

**Files:**
- Create: `/home/drift/ts3/SF-v1/CineBrain/configs/sf_v1/sf_v1_stage2_fusion.yaml`

**Step 1: Create the config**

```yaml
args:
  checkpoint_activations: true
  eval_batch_size: 1
  eval_interval: 500
  eval_iters: 1
  experiment_name: sf_v1_stage2_fusion
  force_train: true
  load: ckpts_5b/sf_v1_p1_full_v2-04-03-13-44
  log_interval: 20
  mode: finetune
  model_parallel_size: 1
  no_load_rng: true
  num_workers: 2
  only_log_video_latents: true
  save: ckpts_5b
  save_interval: 500
  split: 1,0,0
  train_data:
  - __LOCAL_CONFIG_DATASET_ROOT__/sub-0005_train_va.json
  train_iters: 2000
  valid_data:
  - __LOCAL_CONFIG_DATASET_ROOT__/sub-0005_test_va.json
data:
  params:
    fps: 8
    max_num_frames: 33
    skip_frms_num: 0
    video_size:
    - 480
    - 720
    sf_target_keys:
    - keyframe_img_emb
    - scene_text_emb
    - flow_token_pca
    - dyn_class_3
    - motion_dir_8
    - ofs_log_zscore
    - temporal_frame_embs
    - flow_mag_traj
  target: data_video.BrainDataset
deepspeed:
  activation_checkpointing:
    contiguous_memory_optimization: false
    partition_activations: false
  bf16:
    enabled: true
  fp16:
    enabled: false
  gradient_accumulation_steps: 2
  gradient_clipping: 0.1
  hysteresis: 2
  loss_scale: 0
  loss_scale_window: 400
  min_loss_scale: 1
  optimizer:
    params:
      betas:
      - 0.9
      - 0.95
      eps: 1e-8
      lr: 0.0001
      weight_decay: 1e-4
    type: Adam
  steps_per_print: 50
  train_micro_batch_size_per_gpu: 1
  wall_clock_breakdown: false
  zero_allow_untested_optimizer: true
  zero_optimization:
    allgather_bucket_size: 1000000000
    contiguous_gradients: false
    cpu_offload: false
    load_from_fp32_weights: false
    overlap_comm: true
    reduce_bucket_size: 1000000000
    reduce_scatter: true
    stage: 2
model:
  # Freeze everything except fusion modules
  not_trainable_prefixes:
  - all
  freeze_slow_branch: true
  freeze_fast_branch: true
  unfreeze_fusion: true
  loss_fn_config:
    params:
      training_stage: fusion
      sf_loss_config:
        lambda_distill_cls: 0.2
        lambda_distill_spatial: 0.2
        lambda_temporal_delta: 1.0
        lambda_temporal_abs: 0.2
        lambda_flow_traj: 0.3
        lambda_dyn: 0.1
```

Key differences from P1 config:
- `load`: points to P1 v2 checkpoint
- `train_iters`: 2000 (shorter)
- `save_interval`: 500 (more frequent)
- `not_trainable_prefixes: [all]` + `unfreeze_fusion: true`
- `training_stage: fusion` (enables L_diff + L_guide)

---

## Task 4: Verify conditioner is_trainable=True

**Files:**
- Check: `/home/drift/ts3/SF-v1/CineBrain/configs/sf_v1/cinebrain_sf_v1_model.yaml`

**Step 1: Confirm is_trainable**

In the model config's conditioner section, the SFBrainEmbedder must have `is_trainable: true`. This controls whether `GeneralConditioner` runs the embedder under `nullcontext` (gradient preserved) vs `torch.no_grad()` (gradient killed).

Search for `is_trainable` in the model yaml. If it's `false` or missing, change to `true`.

**This is critical** — if `is_trainable=false`, gradient from L_diff cannot reach GatedFusion regardless of all other settings.

---

## Task 5: Preflight verification

**Step 1: Run preflight check**

```bash
ssh ts3 "ssh gpu2 'cd /public/home/maoyaoxin/zhangt/xxt/SF-v1/CineBrain && \
  CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=0 \
  /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python tools/preflight_check.py'"
```

Expected: All checks PASS, forward pass CLEAN.

**Step 2: Verify freeze/unfreeze with a quick script**

Create a one-off check: after model init, print all parameters with `requires_grad=True`. Confirm only `gated_fusion.*` and `guidance_adapter.*` are trainable.

Can reuse `verify_gradient_passthrough.py` or add a print loop in diffusion_video_brain.py debug output.

---

## Task 6: Overfit test (1 sample, 200 steps)

**Step 1: Create overfit config**

Copy `sf_v1_stage2_fusion.yaml` to `sf_v1_stage2_overfit.yaml`, change:
- `train_iters: 200`
- `eval_interval: 50`
- `experiment_name: sf_v1_stage2_overfit`
- Use mini dataset or just train on first sample

**Step 2: Run overfit**

```bash
ssh ts3 "ssh gpu2 'cd /public/home/maoyaoxin/zhangt/xxt/SF-v1/CineBrain && \
  CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=3 \
  nohup /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python \
  train_video_fmri.py --base configs/sf_v1/cinebrain_sf_v1_model.yaml configs/sf_v1/sf_v1_stage2_overfit.yaml \
  --seed 42 > logs/stage2_overfit.log 2>&1 &'"
```

**Step 3: Check results**

- L_diff should decrease from its initial value
- GatedFusion grad norms logged should be non-zero (if gradient logging is in place)
- If L_diff does NOT decrease → gradient vanishing through 42-layer DiT, need fallback (auxiliary context loss)

---

## Task 7: Mini train (mini500, 500 steps)

Only proceed if Task 6 overfit succeeds.

**Step 1: Run mini train**

Use the full Stage 2 config but with `train_iters: 500` and mini dataset.

**Step 2: Check**

- L_diff should show downward trend
- Gating weights should start differentiating (run `evaluate_p1.py` to check α_mot high/low dyn gap)

---

## Task 8: Full Stage 2 training

Only proceed if Task 7 mini train shows L_diff decreasing.

**Step 1: Launch full training**

```bash
ssh ts3 "ssh gpu2 'cd /public/home/maoyaoxin/zhangt/xxt/SF-v1/CineBrain && \
  CUDA_HOME=/usr/local/cuda-12.4 \
  CUDA_VISIBLE_DEVICES=3,4,5,6,7 \
  NCCL_TIMEOUT=3600 \
  nohup /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python \
  -m torch.distributed.run --standalone --nproc_per_node=5 \
  train_video_fmri.py --base configs/sf_v1/cinebrain_sf_v1_model.yaml configs/sf_v1/sf_v1_stage2_fusion.yaml \
  --seed 42 > logs/stage2_fusion.log 2>&1 &'"
```

**Step 2: Monitor**

- Watch `logs/stage2_fusion.log` for L_diff, sf/L_slow, sf/L_fast breakdown
- Key signal: L_diff should steadily decrease

**Step 3: Evaluate**

After 2000 iter, run `tools/evaluate_p1.py` on the Stage 2 checkpoint. Compare gating behavior to P1 baseline. α_mot high/low dyn gap should be larger.

---

## Fallback: If DiT gradient vanishes (Task 6 fails)

If overfit test shows L_diff not decreasing (gradient vanishes through 42-layer DiT):

**Option A: Add auxiliary context loss**

In `sf_losses.py`, add a direct loss on the context tensor:
```python
# Auxiliary context alignment (fallback if DiT gradient vanishes)
if self.training_stage == "fusion" and self.lambda_context > 0:
    # Use video latent mean as rough target for context
    context_pooled = cond["crossattn"].mean(dim=1)  # (B, 4096)
    target_pooled = input.mean(dim=(1,2,3,4))  # rough video signal
    # ... project dimensions and compute MSE
```

**Option B: Skip Stage 2, go directly to Stage 3 (joint)**

Set `training_stage: joint`, unfreeze DiT with very small lr (1e-6), larger lr for fusion modules (1e-4). Use optimizer parameter groups.
