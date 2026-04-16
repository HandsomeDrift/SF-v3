# Brain-Only Recovery Design

**Date:** 2026-04-11

## Goal
- Recover usable non-brain guidance behavior without restarting the full project pipeline.
- Avoid expensive blind retraining by adding low-cost validation before any long Stage 2/3 run.

## Root Cause Summary
- The active Stage 3 rerun inherited `ckpts_5b/phase1v2_s2_04-09-16-37-04-09-16-38/2000`, and `evaluate_p1` confirms that checkpoint was already fully brain-only.
- The previous reset only reinitialized `gate_net` and guidance `out_proj`, leaving the rest of fusion/guidance in a degenerate inherited state.
- `GuidanceLoss` supervises branch heads but does not supervise `alphas` or fused guidance behavior, so training does not directly penalize the `brain-only` shortcut.
- `MultiGuidanceAdapter` currently duplicates the brain latent path by using both `context = z_b` and `alpha_brain * z_b`, which makes `alpha_brain` the easiest route for optimization.

## Design Decisions

### 1. Normalize guidance weights
- Change fusion gating from independent sigmoid gates to normalized weights.
- `gate_net` will output logits and fusion will convert them with `softmax` into `alpha_key/txt/mot/brain`.
- Resetting the last linear to zero should therefore produce a uniform prior of `0.25` each instead of `0.5` each.

### 2. Remove duplicated brain shortcut
- Keep the fused latent `z_b` as one weighted channel instead of using it both as the base path and as an extra gated residual.
- New adapter composition:
  - `context = alpha_brain * z_b`
  - `+ alpha_key * key_guidance`
  - `+ alpha_txt * text_guidance`
  - `+ alpha_mot * motion_guidance`
- This preserves `alpha_brain` for diagnostics and evaluation while eliminating the duplicated shortcut.

### 3. Add direct anti-collapse supervision
- Extend `GuidanceLoss` with gating-aware terms.
- New terms:
  - `L_alpha_mot_dyn`: BCE between `alpha_mot` and `gt_dyn_label_2class`
  - `L_alpha_nonbrain`: hinge-style loss enforcing a minimum non-brain mass `alpha_key + alpha_txt + alpha_mot`
- These terms must be lightweight and only strong enough to prevent collapse, not force uniform alphas.

### 4. Support full fusion/guidance reinitialization
- Add explicit full-reset methods for:
  - fusion projections, fusion cross-attention stack, fusion output projection, and gate head
  - guidance projections and guidance cross-attention projections
- Keep the existing partial reset path for compatibility, but add a new config-controlled full reset for Stage 2.5 recovery experiments.

### 5. Validate before long training
- Add a small test script covering the new normalized gating and gating loss behavior.
- Add a lightweight smoke workflow for Stage 2.5 recovery:
  - start from a healthy Stage 1/P1 checkpoint
  - full-reset only fusion/guidance
  - run a short `200-500` step smoke experiment
  - run `evaluate_p1` at `200` and `500`
- Only proceed to a longer Stage 2.5 if the smoke run exits the `brain-only` state.

## Training Strategy
- Do not reuse the collapsed `phase1v2_s2/2000` checkpoint as the parent for recovery.
- Use the healthy Stage 1/P1 checkpoint as the parent for Stage 2.5 recovery.
- Freeze slow/fast branches and train only fusion/guidance first.
- Promote to Stage 3 only after `evaluate_p1` confirms non-zero `alpha_key/txt/mot` and defined motion-gating correlation.

## Success Criteria
- Unit test verifies normalized alphas, reset behavior, and anti-collapse loss terms.
- Smoke run shows:
  - `alpha_key/txt/mot` are not all zero
  - `alpha_brain` is no longer the only active channel
  - `gating_alpha_mot_dyn_spearman` becomes defined
- If these conditions fail in the short smoke run, do not start a full Stage 2.5 training job.
