# Stage 3 Training Debug Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the Stage 3 training startup failure so the focal conservative config can resume from the intended checkpoint and progress past iteration 0.

**Architecture:** First fix checkpoint-restore semantics so `1000` can be loaded explicitly without producing `.../1000/1000/...`. Then add a reusable single-step multi-rank diagnostic harness to isolate the exact failing boundary in the first training step (`forward`, `all_reduce`, `backward`, or `optimizer step`). Finally, apply the smallest code change needed at the failing boundary and verify the real training job advances beyond iteration 0.

**Tech Stack:** PyTorch Distributed (`torchrun`), DeepSpeed ZeRO-2, SAT training framework, YAML config loading, CineBrain Stage 3 training pipeline.

---

### Task 1: Add explicit checkpoint iteration override

**Files:**
- Modify: `/home/drift/ts3/SF-v1/CineBrain/arguments.py`
- Modify: `/home/drift/ts3/SF-v1/CineBrain/sat/training/deepspeed_training.py:96-99`
- Reference: `/home/drift/ts3/SF-v1/CineBrain/sat/training/model_io.py:270-297`
- Test: single-process load command on `/home/drift/ts3/SF-v1/CineBrain`

**Step 1: Write the failing reproduction command**

Run this command and confirm it fails with the duplicated path:

```bash
ssh ts3 "ssh gpu3 'cd /public/home/maoyaoxin/xxt/SF-v1/CineBrain && \
  CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29696 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
  /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python tmp_diag_single_forward.py'"
```

Expected: FAIL when config uses `load: .../1000` because SAT builds `.../1000/1000/mp_rank_00_model_states.pt`.

**Step 2: Add a dedicated CLI/config arg for explicit iteration**

Add a parser arg in `arguments.py`:

```python
group.add_argument("--load-iteration", type=int, default=None,
                   help="explicit checkpoint iteration to load from args.load parent directory")
```

**Step 3: Thread the override into training_main**

Change checkpoint loading in `sat/training/deepspeed_training.py` from:

```python
args.iteration = load_checkpoint(model, args)
```

to:

```python
args.iteration = load_checkpoint(
    model,
    args,
    specific_iteration=getattr(args, "load_iteration", None),
)
```

**Step 4: Keep YAML `load` pointing at the parent directory**

Use:

```yaml
load: ckpts_5b/sf_v1_stage3_full_recovery_resume_v3-04-14-19-18
```

and set the desired restore point with:

```yaml
load_iteration: 1000
```

(or via CLI).

**Step 5: Verify explicit 1000 restore works**

Run:

```bash
ssh ts3 "ssh gpu3 'cd /public/home/maoyaoxin/xxt/SF-v1/CineBrain && \
  CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29697 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 \
  /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python tmp_diag_single_forward.py \
  --load-iteration 1000'"
```

Expected: PASS, and logs should show loading:

```text
.../sf_v1_stage3_full_recovery_resume_v3-04-14-19-18/1000/mp_rank_00_model_states.pt
```

**Step 6: Commit**

```bash
git -C /home/drift/ts3/SF-v1/CineBrain add arguments.py sat/training/deepspeed_training.py configs/sf_v1/sf_v1_stage3_joint_focal_conservative.yaml
git -C /home/drift/ts3/SF-v1/CineBrain commit -m "fix: support explicit checkpoint iteration restore"
```

---

### Task 2: Promote the single-step training diagnostic into a reusable tool

**Files:**
- Create: `/home/drift/ts3/SF-v1/CineBrain/tools/debug_stage3_train_step.py`
- Reference: `/home/drift/ts3/SF-v1/CineBrain/tmp_diag_train_step.py`
- Reference: `/home/drift/ts3/SF-v1/CineBrain/train_video_fmri.py:201-225`
- Reference: `/home/drift/ts3/SF-v1/CineBrain/sat/training/deepspeed_training.py:428-542`

**Step 1: Write the failing diagnostic harness contract**

The tool must print these boundaries on every rank:

```python
print(f"rank {rank} before forward_step")
print(f"rank {rank} after forward_step")
print(f"rank {rank} before loss all_reduce")
print(f"rank {rank} after loss all_reduce")
print(f"rank {rank} before backward")
print(f"rank {rank} after backward")
print(f"rank {rank} before step")
print(f"rank {rank} after step")
```

**Step 2: Implement the reusable script**

Move the successful parts of `tmp_diag_train_step.py` into `tools/debug_stage3_train_step.py`.

The script should:
- parse normal training args via `get_args()`
- build the model via `get_model()`
- restore checkpoint via `load_checkpoint(..., specific_iteration=args.load_iteration)`
- build deepspeed model/optimizer via `setup_model_untrainable_params_and_optimizer()`
- build loaders via `make_loaders()`
- execute exactly one training step boundary-by-boundary

**Step 3: Run the diagnostic on an alternate node with free GPUs**

Example command:

```bash
ssh ts3 "ssh gpu3 'cd /public/home/maoyaoxin/xxt/SF-v1/CineBrain && \
  CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=0,2,3 \
  /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/torchrun \
  --standalone --nproc_per_node=3 tools/debug_stage3_train_step.py \
  --base configs/sf_v1/cinebrain_sf_v1_model.yaml configs/sf_v1/sf_v1_stage3_joint_focal_conservative.yaml \
  --load-iteration 1000 --seed 42'"
```

Expected: one of two outcomes:
- PASS through `after step` on all ranks
- FAIL/hang at a specific boundary that identifies the failing subsystem

**Step 4: Save the boundary result**

Record the last successful boundary in the handoff/log notes before making any fix.

**Step 5: Commit**

```bash
git -C /home/drift/ts3/SF-v1/CineBrain add tools/debug_stage3_train_step.py
git -C /home/drift/ts3/SF-v1/CineBrain commit -m "chore: add stage3 single-step training diagnostic"
```

---

### Task 3: Fix the failing first-step boundary with the smallest possible code change

**Files:**
- Modify exactly one of:
  - `/home/drift/ts3/SF-v1/CineBrain/train_video_fmri.py:201-225`
  - `/home/drift/ts3/SF-v1/CineBrain/sat/training/deepspeed_training.py:442-518`
  - `/home/drift/ts3/SF-v1/CineBrain/diffusion_video_brain.py:232-260`
- Test: `/home/drift/ts3/SF-v1/CineBrain/tools/debug_stage3_train_step.py`

**Step 1: Use the diagnostic output to write the failing condition**

Examples:
- If last log is `after forward_step` but never `after loss all_reduce`, the failure is in distributed reduction.
- If last log is `after loss all_reduce` but never `after backward`, the failure is in backward.
- If last log is `after backward` but never `after step`, the failure is in `model.step()` / optimizer path.

Write the observed failing boundary into your notes before changing code.

**Step 2: Implement the minimal fix at the failing boundary**

Examples of acceptable minimal fixes:
- make a scalar metric reduction robust to missing/invalid tensors
- guard first-step logging/metric reduction for a non-scalar value
- fix a bad optimizer-step assumption under ZeRO-2 + gradient accumulation
- fix a rank-local branch that diverges before collective ops

Do **not** change unrelated model architecture or loss design.

**Step 3: Re-run the single-step diagnostic**

Run the same command from Task 2.

Expected: PASS through:

```text
before step
after step
success
```

on every rank.

**Step 4: Commit**

```bash
git -C /home/drift/ts3/SF-v1/CineBrain add train_video_fmri.py sat/training/deepspeed_training.py diffusion_video_brain.py
# stage only the file(s) actually modified
git -C /home/drift/ts3/SF-v1/CineBrain commit -m "fix: unblock stage3 first training step"
```

---

### Task 4: Verify a real short training run advances past iteration 0

**Files:**
- Modify: `/home/drift/ts3/SF-v1/CineBrain/configs/sf_v1/sf_v1_stage3_joint_focal_conservative.yaml`
- Log: `/public/home/maoyaoxin/xxt/SF-v1/CineBrain/logs/focal_conservative_test.log`

**Step 1: Keep the short-run config conservative**

Use:

```yaml
train_iters: 300
eval_interval: 100
save_interval: 100
num_workers: 2
load: ckpts_5b/sf_v1_stage3_full_recovery_resume_v3-04-14-19-18
load_iteration: 1000
```

**Step 2: Start the short run on a node with confirmed free GPUs**

Example:

```bash
ssh ts3 "ssh gpu3 'cd /public/home/maoyaoxin/xxt/SF-v1/CineBrain && \
  CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=0,2,3 \
  nohup /public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/torchrun \
  --standalone --nproc_per_node=3 train_video_fmri.py \
  --base configs/sf_v1/cinebrain_sf_v1_model.yaml configs/sf_v1/sf_v1_stage3_joint_focal_conservative.yaml \
  --seed 42 > logs/focal_conservative_test.log 2>&1 &'"
```

**Step 3: Verify it really moved past startup**

Run:

```bash
ssh ts3 "ssh gpu3 'grep -E \"iteration|total loss|L_router|lambda_router\" /public/home/maoyaoxin/xxt/SF-v1/CineBrain/logs/focal_conservative_test.log | tail -20'"
```

Expected: at least one real training iteration / loss line beyond the startup argument dump.

**Step 4: Commit config changes if needed**

```bash
git -C /home/drift/ts3/SF-v1/CineBrain add configs/sf_v1/sf_v1_stage3_joint_focal_conservative.yaml
git -C /home/drift/ts3/SF-v1/CineBrain commit -m "fix: restore stage3 conservative run from explicit 1000 checkpoint"
```

---

### Task 5: Hand the verified fix back to the real target run on gpu2

**Files:**
- Modify if needed: `/home/drift/fitten/SF-v1/HANDOFF.md`
- Verify: `/public/home/maoyaoxin/xxt/SF-v1/CineBrain/logs/focal_conservative_test.log`

**Step 1: Re-check gpu2 availability before using it**

Run:

```bash
ssh ts3 "ssh gpu2 'nvidia-smi && ps aux | grep python | grep -v grep'"
```

Expected: only proceed if the intended GPUs are free and no other user process is disturbed.

**Step 2: Start the real target training with the verified fix**

Use the same launch pattern as the successful short run, adapted to gpu2 and the desired GPU set.

**Step 3: Verify the training is alive**

Check both:

```bash
ssh ts3 "ssh gpu2 'pgrep -af \"torchrun|train_video_fmri.py\"'"
ssh ts3 "ssh gpu2 'tail -100 /public/home/maoyaoxin/xxt/SF-v1/CineBrain/logs/focal_conservative_test.log'"
```

Expected:
- active torchrun/train processes exist
- log contains real training-step output, not just startup argument dump

**Step 4: Update handoff**

Record:
- the actual root cause
- the exact fix
- the verified launch command
- current run status and next check point

**Step 5: Commit final code changes**

```bash
git -C /home/drift/ts3/SF-v1/CineBrain status
git -C /home/drift/ts3/SF-v1/CineBrain add <exact modified files>
git -C /home/drift/ts3/SF-v1/CineBrain commit -m "fix: restore stage3 training startup"
```
