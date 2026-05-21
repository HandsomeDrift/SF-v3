# Gating Semantics Recovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current motion-heavy but semantics-insensitive gate with a hierarchy that separates brain-vs-guidance allocation from semantic-vs-motion allocation, then validate the change with low-cost tests before any longer Stage 2.5 run.

**Architecture:** The fusion gate will stop using a flat 4-way competition and instead predict three decisions: how much budget leaves the base brain path, how much of that non-brain budget goes to motion versus semantics, and how semantic budget splits between key/text. The guidance adapter will let motion attend over temporal tokens rather than only a pooled EEG summary, and the loss will add relative routing supervision so dynamic clips prefer motion while static clips prefer semantic guidance.

**Tech Stack:** PyTorch, current CineBrain-SF encoders/losses, lightweight Python test scripts, existing Stage 2.5 smoke/eval workflow.

---

### Task 1: Write the failing tests for hierarchical gating and temporal motion guidance

**Files:**
- Modify: `/home/drift/ts3/SF-v1/CineBrain/tools/test_brain_only_recovery.py`

**Step 1: Write the failing test**
- Add one test that expects `reset_gate_net()` to restore a uniform `0.25 / 0.25 / 0.25 / 0.25` prior under the new hierarchical parameterization.
- Add one test that expects `temporal_tokens` to change the motion-guided output when `alpha_mot` is active and pooled EEG features are held fixed.
- Add one test that expects a dynamic clip with low `alpha_mot` or a static clip with high `alpha_mot` to incur a positive routing penalty.

**Step 2: Run test to verify it fails**

Run:
```bash
ssh ts3 "cd /public/home/maoyaoxin/zhangt/xxt/SF-v1/CineBrain && /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python tools/test_brain_only_recovery.py"
```

Expected:
- failure because the current code still uses a flat 4-way gate, ignores `temporal_tokens` inside `mot_attn`, and lacks routing-margin supervision

### Task 2: Implement hierarchical gate budgeting

**Files:**
- Modify: `/home/drift/ts3/SF-v1/CineBrain/sgm/modules/encoders/gated_fusion.py`

**Step 1: Write minimal implementation**
- Keep `gate_net` output size at `4`, but reinterpret the outputs as:
  - `nonbrain_logit`
  - `motion_share_logit`
  - `semantic_key_logit`
  - `semantic_txt_logit`
- Convert logits into:
  - `alpha_brain = 1 - sigmoid(nonbrain_logit)`
  - `nonbrain = sigmoid(nonbrain_logit)`
  - `motion_share = sigmoid(motion_share_logit)`
  - `semantic_share = 1 - motion_share`
  - `key_txt = softmax([semantic_key_logit, semantic_txt_logit])`
  - `alpha_mot = nonbrain * motion_share`
  - `alpha_key = nonbrain * semantic_share * key_txt[0]`
  - `alpha_txt = nonbrain * semantic_share * key_txt[1]`
- Update `reset_gate_net()` so the zero-weight prior still produces exactly `0.25` for all four alphas.

**Step 2: Re-run the failing test**
- Confirm the hierarchical-prior test passes while the temporal-motion and routing-margin tests stay red

### Task 3: Feed temporal tokens into motion guidance

**Files:**
- Modify: `/home/drift/ts3/SF-v1/CineBrain/sgm/modules/encoders/multi_guidance.py`

**Step 1: Write minimal implementation**
- Keep `z_b` as the base path.
- Make motion guidance attend over a small sequence:
  - summary token from `eeg_pooled_proj`
  - temporal token sequence from `fast_out["temporal_tokens"]` when available
  - fallback to `global_dyn_token` only when temporal tokens are absent
- Keep key/text guidance unchanged.
- Preserve existing reset helpers.

**Step 2: Re-run the failing test**
- Confirm the temporal-motion test turns green

### Task 4: Add relative routing supervision for motion-vs-semantics

**Files:**
- Modify: `/home/drift/ts3/SF-v1/CineBrain/sgm/modules/diffusionmodules/sf_losses.py`
- Modify: `/home/drift/ts3/SF-v1/CineBrain/sgm/modules/diffusionmodules/loss.py`

**Step 1: Write minimal implementation**
- Extend `GuidanceLoss` with:
  - `lambda_alpha_motion_margin`
  - `alpha_motion_margin`
- Add `L_alpha_motion_margin`:
  - for dynamic clips, penalize `alpha_mot < alpha_key + alpha_txt + margin`
  - for static clips, penalize `alpha_mot > alpha_key + alpha_txt - margin`
- Keep the existing anti-collapse losses:
  - `L_alpha_mot_dyn`
  - `L_alpha_nonbrain`
- Thread and log the new loss term through `VideoDiffusionLossSF`.

**Step 2: Re-run the failing test**
- Confirm the routing-margin test turns green and the full test file passes

### Task 5: Run low-cost regression checks

**Files:**
- Use existing:
  - `/home/drift/ts3/SF-v1/CineBrain/tools/test_brain_only_recovery.py`
  - `/home/drift/ts3/SF-v1/CineBrain/tools/smoke_test_stage2_recovery.py`

**Step 1: Run the unit-style recovery test**

Run:
```bash
ssh ts3 "cd /public/home/maoyaoxin/zhangt/xxt/SF-v1/CineBrain && /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python tools/test_brain_only_recovery.py"
```

Expected:
- all tests pass

**Step 2: Run the smoke test**

Run:
```bash
ssh ts3 "ssh gpu2 'cd /public/home/maoyaoxin/zhangt/xxt/SF-v1/CineBrain && CUDA_VISIBLE_DEVICES=4 /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python tools/smoke_test_stage2_recovery.py --ckpt <healthy-stage1-ckpt> --dataset-json <train-json> --model-config configs/sf_v1/cinebrain_sf_v1_model.yaml --num-steps 5 --full-reset'"
```

Expected:
- temporal tokens influence motion guidance
- fusion/guidance gradients stay non-zero
- alphas do not immediately re-collapse to a single route

### Task 6: Decide whether to relaunch a short Stage 2.5 run

**Files:**
- No code changes required

**Step 1: Decision gate**
- If smoke still shows one-route dominance with no temporal sensitivity, stop and redesign before training.
- If smoke is stable, rerun a short `100-200` iteration Stage 2.5 recovery job and check `evaluate_p1` at the first save point.
