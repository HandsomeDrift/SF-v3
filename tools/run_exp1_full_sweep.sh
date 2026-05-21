#!/bin/bash
# §6 Exp 1 — Single-step α_brain perturbation sweep (FULL, not pilot).
# Uses v2 static checkpoint as baseline.
#
# Sweeps i* ∈ {1, 7, 13, 20, 27, 34, 41, 48} (8 τ values),
# with N_SAMPLES samples per step + 1 baseline (shared across steps).
# Each perturb trajectory uses δα_brain = 0.3.
#
# Parallelism: 2 GPUs on gpu2 (GPU 0 + GPU 1), configs interleaved.
#
# Preconditions: run_exp1_unit_test.sh passed first (T1/T3/T4 all green).
# Cost: 8 step configs × N_SAMPLES samples ÷ 2 GPUs × 4.5 min/sample.
#       With N_SAMPLES=50: 900 min on 2 GPUs = 15 hours wall-clock.
#       With N_SAMPLES=20: 360 min on 2 GPUs = 6 hours wall-clock.
#       With N_SAMPLES=10 (pilot): 180 min on 2 GPUs = 3 hours wall-clock.

set -eo pipefail

cd /public/home/maoyaoxin/zhangt/xxt/SF-v3

# ---- Configurable ----
N_SAMPLES=${N_SAMPLES:-10}       # override via env var
PERTURB_DELTA=${PERTURB_DELTA:-0.3}
PORT_BASE=${PORT_BASE:-29910}
OUTPUT_BASE=${OUTPUT_BASE:-results/exp1_alpha_sweep}
# Steps (indices into sampler's 50 discretization steps).
# Avoid index 0 and 49 (endpoints) which can have edge-case behavior.
STEPS=(1 7 13 20 27 34 41 48)

# ---- Setup ----
DATASET_JSON=/public/home/maoyaoxin/zhangt/xxt/datasets/exp1_sweep_${N_SAMPLES}samples.json
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python

if [ ! -f "${DATASET_JSON}" ]; then
  echo "Generating ${N_SAMPLES}-sample dataset JSON from mini20..."
  ${PY} <<EOF
import json, random
src = json.load(open("/public/home/maoyaoxin/zhangt/xxt/datasets/sub-0005_test_va.json"))
random.seed(42)
subset = random.sample(src, k=${N_SAMPLES})
json.dump(subset, open("${DATASET_JSON}", "w"))
print(f"Wrote {len(subset)} samples to ${DATASET_JSON}")
EOF
fi

mkdir -p ${OUTPUT_BASE}

# ---- Launch baseline once on GPU 0 (shared across all step comparisons) ----
BASELINE_DIR=${OUTPUT_BASE}/baseline
if [ ! -f "${BASELINE_DIR}/done.flag" ]; then
  echo "=== Launching baseline on GPU 0 ==="
  mkdir -p ${BASELINE_DIR}
  CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/public/home/maoyaoxin/zhangt/xxt/SF-v3 \
    nohup ${PY} -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=${PORT_BASE} \
    sample_brain_va.py --base \
      configs/sf_v1/cinebrain_sf_v1_model.yaml \
      configs/sf_v1/infer_stage3_v2.yaml \
    --seed 42 \
    --jsonpath ${DATASET_JSON} \
    --output_dir ${BASELINE_DIR} \
    --save_latents \
    > ${OUTPUT_BASE}/baseline.log 2>&1 &
  BASELINE_PID=$!
  echo "Baseline PID=${BASELINE_PID}, waiting to complete..."
  wait ${BASELINE_PID}
  touch ${BASELINE_DIR}/done.flag
fi

# ---- Loop through steps, alternating GPUs 0/1 ----
PORT=${PORT_BASE}
for idx in "${!STEPS[@]}"; do
  step=${STEPS[${idx}]}
  gpu=$((idx % 2))
  PORT=$((PORT + 1))

  out_dir=${OUTPUT_BASE}/step_${step}
  override_yaml=/tmp/exp1_sweep_override_step${step}.yaml

  if [ -f "${out_dir}/done.flag" ]; then
    echo "Step ${step} already done, skipping"
    continue
  fi

  mkdir -p ${out_dir}
  sed "s/PERTURB_STEP/${step}/g; s/PERTURB_DELTA/${PERTURB_DELTA}/g" \
    configs/sf_v1/exp1_perturb_override.yaml > ${override_yaml}

  echo "=== Launching step=${step} on GPU ${gpu}, port ${PORT} ==="
  CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=/public/home/maoyaoxin/zhangt/xxt/SF-v3 \
    nohup ${PY} -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=${PORT} \
    sample_brain_va.py --base \
      configs/sf_v1/cinebrain_sf_v1_model.yaml \
      ${override_yaml} \
      configs/sf_v1/infer_stage3_v2.yaml \
    --seed 42 \
    --jsonpath ${DATASET_JSON} \
    --output_dir ${out_dir} \
    --save_latents \
    > ${OUTPUT_BASE}/step_${step}.log 2>&1 &

  # Launch two configs in parallel (one per GPU), then wait both
  if [ $((idx % 2)) -eq 1 ]; then
    wait
    echo "  Batch complete: steps ${STEPS[$((idx-1))]}, ${step}"
    touch ${OUTPUT_BASE}/step_${STEPS[$((idx-1))]}/done.flag
    touch ${out_dir}/done.flag
  fi
done

# Wait for any trailing single job
wait

# Mark any remaining steps done
for step in "${STEPS[@]}"; do
  touch ${OUTPUT_BASE}/step_${step}/done.flag
done

echo ""
echo "=== Exp 1 sweep complete ==="
echo "Run analysis:"
echo "  ${PY} tools/analyze_ood_asymmetry.py --exp exp1 --base ${OUTPUT_BASE}"
