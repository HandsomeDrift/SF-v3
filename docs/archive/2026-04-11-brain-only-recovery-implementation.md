# Brain-Only Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current collapse-prone fusion/guidance behavior with normalized gating, direct anti-collapse supervision, and a low-cost Stage 2.5 recovery workflow.

**Architecture:** Fusion will emit normalized alpha weights, the guidance adapter will stop duplicating the brain latent path, and the loss stack will directly supervise `alpha_mot` and minimum non-brain usage. Recovery will start from a healthy Stage 1 parent and use a short smoke run before any long training.

**Tech Stack:** PyTorch, existing CineBrain-SF training stack, YAML configs, shell launch scripts, lightweight Python smoke tests.

---

### Task 1: Add failing recovery tests

**Files:**
- Create: `/home/drift/ts3/SF-v1/CineBrain/tools/test_brain_only_recovery.py`

**Step 1: Write the failing test**
- Add three checks:
  - fusion alphas sum to 1.0 across the four channels
  - `reset_gate_net()` restores a uniform `0.25` prior
  - `GuidanceLoss` exposes positive anti-collapse penalties for `brain-only` alphas

**Step 2: Run test to verify it fails**

Run:
```bash
ssh ts3 "cd /public/home/maoyaoxin/zhangt/xxt/SF-v1/CineBrain && /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python tools/test_brain_only_recovery.py"
```

Expected:
- failure because fusion uses sigmoid gates and `GuidanceLoss` has no alpha-aware penalties yet

### Task 2: Normalize fusion gating and add full reset support

**Files:**
- Modify: `/home/drift/ts3/SF-v1/CineBrain/sgm/modules/encoders/gated_fusion.py`

**Step 1: Implement minimal code**
- Remove the final sigmoid from `gate_net`
- Convert gate logits to `softmax` weights in `forward`
- Keep output dict keys unchanged for compatibility
- Add a `reset_fusion_state(full=False)` method
- Preserve `reset_gate_net()` but make it restore a uniform `0.25` prior

**Step 2: Re-run the failing test**
- Confirm the normalized alpha tests now pass while the alpha-loss test still fails

### Task 3: Remove the duplicated brain path and add full guidance reset

**Files:**
- Modify: `/home/drift/ts3/SF-v1/CineBrain/sgm/modules/encoders/multi_guidance.py`

**Step 1: Implement minimal code**
- Stop using both `context = z_b` and `alpha_brain * z_b`
- Build `context` as the weighted sum of brain/key/text/motion channels
- Remove the old alpha floor path
- Add `reset_guidance_state(full=False)` and keep `reset_guidance_outputs()` as the partial reset API

**Step 2: Re-run the failing test**
- Confirm fusion reset tests still pass and the alpha-loss test remains the only red item

### Task 4: Add anti-collapse alpha supervision

**Files:**
- Modify: `/home/drift/ts3/SF-v1/CineBrain/sgm/modules/diffusionmodules/sf_losses.py`
- Modify: `/home/drift/ts3/SF-v1/CineBrain/sgm/modules/diffusionmodules/loss.py`

**Step 1: Implement minimal code**
- Extend `GuidanceLoss` constructor with:
  - `lambda_alpha_mot`
  - `lambda_alpha_nonbrain`
  - `alpha_nonbrain_floor`
- Extend `GuidanceLoss.forward(...)` to accept `alphas`
- Add:
  - BCE(`alpha_mot`, `gt_dyn_label_2class`)
  - `relu(alpha_nonbrain_floor - (alpha_key + alpha_txt + alpha_mot))`
- Thread `embedder._last_alphas` through `VideoDiffusionLossSF`
- Log the new alpha-loss terms in the loss breakdown

**Step 2: Re-run the failing test**
- Confirm all recovery tests pass

### Task 5: Add full-reset training support

**Files:**
- Modify: `/home/drift/ts3/SF-v1/CineBrain/diffusion_video_brain.py`

**Step 1: Implement minimal code**
- Add model config flag `reset_fusion_guidance_full`
- If enabled, call the new full reset methods after checkpoint load
- Keep the old `reset_gate_net` behavior working for existing scripts

**Step 2: Verify**
- Run the recovery tests again

### Task 6: Add a low-cost recovery smoke tool

**Files:**
- Create: `/home/drift/ts3/SF-v1/CineBrain/tools/smoke_test_stage2_recovery.py`

**Step 1: Implement minimal code**
- Load a healthy Stage 1/P1 checkpoint
- Apply full fusion/guidance reset
- Run a few forward/backward steps on fusion/guidance only
- Print:
  - alpha means before/after a few optimizer steps
  - gradient norms for fusion/guidance
  - anti-collapse loss terms

**Step 2: Verify**

Run:
```bash
ssh ts3 "ssh gpu2 'cd /public/home/maoyaoxin/zhangt/xxt/SF-v1/CineBrain && CUDA_VISIBLE_DEVICES=4 /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python tools/smoke_test_stage2_recovery.py --num-steps 5'"
```

Expected:
- non-zero gradients on fusion/guidance
- alpha-related losses are non-zero when initialized from a collapsed parent or after forced collapse

### Task 7: Add Stage 2.5 smoke config and launcher

**Files:**
- Create: `/home/drift/ts3/SF-v1/CineBrain/configs/sf_v1/sf_v1_stage2_recovery.yaml`
- Create: `/home/drift/ts3/SF-v1/CineBrain/run_stage2_recovery.sh`

**Step 1: Implement minimal code**
- Start from the Stage 2 fusion config
- Parent checkpoint must be healthy Stage 1/P1, not `phase1v2_s2`
- Enable:
  - normalized gating path
  - alpha-aware loss terms
  - `reset_fusion_guidance_full: true`
- Default to a short smoke run:
  - `train_iters=500`
  - `eval_interval=100`
  - `save_interval=100`

**Step 2: Verify configuration**
- Print the generated YAML in dry-run mode or inspect the saved file

### Task 8: Run the short Stage 2.5 smoke experiment

**Files:**
- Use the new launcher and config

**Step 1: Run**

Run:
```bash
ssh ts3 "ssh gpu2 'cd /public/home/maoyaoxin/zhangt/xxt/SF-v1/CineBrain && NGPU=1 TRAIN_ITERS=500 EVAL_INTERVAL=100 SAVE_INTERVAL=100 bash run_stage2_recovery.sh'"
```

**Step 2: Evaluate**
- Run `tools/evaluate_p1.py` on the `200` and `500` checkpoints
- Stop immediately if the run is still `brain-only`

### Task 9: Decide whether to scale up

**Files:**
- No code changes required

**Step 1: Decision gate**
- If `alpha_key/txt/mot` remain zero: stop and redesign
- If non-brain alphas recover: scale the same config to a longer Stage 2.5 run
- Only then promote the recovered Stage 2.5 checkpoint to Stage 3
