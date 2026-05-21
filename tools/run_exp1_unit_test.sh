#!/bin/bash
# §6 Exp 1 — unit tests for perturbation hooks (T1 bit-identical, T3 non-zero, T4 monotonicity).
#
# Preconditions:
#   - sampling.py已经加了 perturb_spec 参数 (2026-04-19)
#   - sample_brain_va.py 已经加了 --save_latents flag
#   - configs/sf_v1/exp1_perturb_override.yaml 存在
#   - datasets/exp1_unit_1sample.json 存在 (1 样本)
#
# Runs 4 configs × 1 sample = ~18 min wall-clock on 1 GPU (GPU 0 of gpu2).
# Path B is using GPU 3 and GPU 5, so this doesn't interfere.

set -eo pipefail

cd /public/home/maoyaoxin/zhangt/xxt/SF-v3

DATASET_JSON=/public/home/maoyaoxin/zhangt/xxt/datasets/exp1_unit_1sample.json
BASE_DIR=results/exp1_unit_test
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
GPU=${GPU:-0}

export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONPATH=/public/home/maoyaoxin/zhangt/xxt/SF-v3

# Force CUDA determinism for bit-reproducible baseline (required for T1 passing).
# Without this, matmul/cuDNN non-det creates ~0.17/element noise floor across runs,
# which swamps perturbation signals of comparable magnitude.
export FORCE_DETERMINISM=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

run_one() {
  local tag=$1
  local override=$2
  local port=$3

  mkdir -p ${BASE_DIR}/${tag}
  echo "=== Running config ${tag} on GPU ${GPU}, port ${port} ==="
  echo "    Override: ${override:-<none (baseline)>}"

  local base_args=(
    "configs/sf_v1/cinebrain_sf_v1_model.yaml"
  )
  if [ -n "${override}" ]; then
    base_args+=("${override}")
  fi
  base_args+=("configs/sf_v1/infer_stage3_v2.yaml")

  ${PY} -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=${port} \
    sample_brain_va.py --base "${base_args[@]}" \
    --seed 42 \
    --jsonpath ${DATASET_JSON} \
    --output_dir ${BASE_DIR}/${tag} \
    --save_latents \
    > ${BASE_DIR}/${tag}.log 2>&1

  echo "    Done: $(ls ${BASE_DIR}/${tag}/*_latent.pt 2>/dev/null | wc -l) latent(s) saved"
}

# Generate override YAMLs via sed
mkdir -p /tmp/exp1_unit_overrides
for spec in "noop_20_0p0:20:0.0" "perturb_early_1_0p3:1:0.3" "perturb_late_48_0p3:48:0.3"; do
  tag="${spec%%:*}"
  rest="${spec#*:}"
  step="${rest%%:*}"
  delta="${rest##*:}"
  sed "s/PERTURB_STEP/${step}/g; s/PERTURB_DELTA/${delta}/g" \
    configs/sf_v1/exp1_perturb_override.yaml \
    > /tmp/exp1_unit_overrides/${tag}.yaml
done

# ── Config A: Baseline (no override, no perturb_spec whatsoever) ──
run_one "A_baseline" "" 29900

# ── Config B: No-op (delta=0.0 → should be BIT-IDENTICAL to A) ──
run_one "B_noop" "/tmp/exp1_unit_overrides/noop_20_0p0.yaml" 29901

# ── Config C: Early perturb (step=1, delta=0.3 → large diff from A) ──
run_one "C_early" "/tmp/exp1_unit_overrides/perturb_early_1_0p3.yaml" 29902

# ── Config D: Late perturb (step=48, delta=0.3 → small diff from A, < C) ──
run_one "D_late" "/tmp/exp1_unit_overrides/perturb_late_48_0p3.yaml" 29903

# ── Analysis ──
echo ""
echo "=========================================="
echo "  Analysis"
echo "=========================================="

${PY} <<'EOF'
import torch, glob, os

def load_latents(dir):
    files = sorted(glob.glob(os.path.join(dir, "*_latent.pt")))
    return {os.path.basename(f): torch.load(f, map_location="cpu") for f in files}

A = load_latents("results/exp1_unit_test/A_baseline")
B = load_latents("results/exp1_unit_test/B_noop")
C = load_latents("results/exp1_unit_test/C_early")
D = load_latents("results/exp1_unit_test/D_late")

if not A or not B or not C or not D:
    print("ERROR: missing latents. Check individual logs.")
    raise SystemExit(1)

def diff(x, y):
    return (x.float() - y.float()).abs().max().item()

def diff_l2(x, y):
    return (x.float() - y.float()).norm().item()

# T1: A vs B (bit-identical, both effectively perturb_spec=noop)
print("T1 BIT-IDENTICAL (A_baseline vs B_noop):")
for k in A:
    d_max = diff(A[k], B[k])
    d_l2 = diff_l2(A[k], B[k])
    print(f"  {k}: max|Δ| = {d_max:.2e}, L2 = {d_l2:.2e}")
t1_max = max(diff(A[k], B[k]) for k in A)
# With FORCE_DETERMINISM=1 we expect bit-identical (0.0); allow 1e-5 slack
# for any residual floating-point corner cases we may have missed.
t1_pass = t1_max < 1e-5
print(f"  OVERALL: max|Δ| = {t1_max:.2e} → {'PASS' if t1_pass else 'FAIL (threshold 1e-5)'}")
print()

# T3: A vs C (non-zero response, perturbed)
print("T3 NON-ZERO RESPONSE (A_baseline vs C_early @ step=1, δ=0.3):")
for k in A:
    d_l2 = diff_l2(A[k], C[k])
    print(f"  {k}: L2 = {d_l2:.2e}")
t3_l2 = sum(diff_l2(A[k], C[k]) for k in A) / len(A)
t3_pass = t3_l2 > 1e-2
print(f"  MEAN L2 = {t3_l2:.2e} → {'PASS' if t3_pass else 'FAIL (threshold 1e-2)'}")
print()

# T4: ||A - C|| >> ||A - D|| (monotonicity smoke)
print("T4 MONOTONICITY (L2(A, C_early) > L2(A, D_late @ step=48, δ=0.3)):")
for k in A:
    dc = diff_l2(A[k], C[k])
    dd = diff_l2(A[k], D[k])
    print(f"  {k}: L2_early = {dc:.2e}, L2_late = {dd:.2e}, ratio = {dc/max(dd,1e-9):.2f}x")
ratio = sum(diff_l2(A[k], C[k]) for k in A) / max(sum(diff_l2(A[k], D[k]) for k in A), 1e-9)
t4_pass = ratio >= 1.5   # early should be ≥ 1.5× larger than late
print(f"  MEAN ratio early/late = {ratio:.2f}x → {'PASS' if t4_pass else 'FAIL (threshold ≥ 1.5x)'}")
print()

# Summary
print("SUMMARY:")
print(f"  T1 bit-identical:  {'PASS' if t1_pass else 'FAIL'}")
print(f"  T3 non-zero:       {'PASS' if t3_pass else 'FAIL'}")
print(f"  T4 monotonicity:   {'PASS' if t4_pass else 'FAIL (this is the go/no-go)'}")
print()

# Go/No-Go decision
if t1_pass and t3_pass and t4_pass:
    print("ALL GREEN: proceed to Exp 1-pilot (10 samples × 8 τ values)")
elif not t1_pass:
    print("BLOCKER: T1 failed → code is not additive; revert sampling.py and debug")
elif not t4_pass:
    print("BLOCKER: T4 failed → theoretical prediction not holding on 1 sample;")
    print("   recheck with larger batch (5 samples) before declaring theory broken")
else:
    print("PARTIAL: investigate which tests failed")
EOF

echo ""
echo "=========================================="
echo "  Logs: ${BASE_DIR}/{A_baseline,B_noop,C_early,D_late}.log"
echo "  Latents: ${BASE_DIR}/*/*_latent.pt"
echo "=========================================="
