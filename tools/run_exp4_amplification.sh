#!/bin/bash
# §6 Exp 4 — Amplification factor A(τ) measurement via single-step LATENT perturbation.
#
# Perturbs x (not C) at a single step with small Gaussian noise η ~ N(0, ε²I),
# then measures ||x_N^pert - x_N^baseline|| / ||η|| as empirical amplification.
# This isolates A(τ) from sensitivity S(τ).
#
# Sweeps same 8 steps as Exp 1. ε is set per Exp 4 pilot result (default 0.01).
#
# Preconditions: run_exp4_pilot.py executed first to select ε; record ε* in env.

set -eo pipefail

cd /public/home/maoyaoxin/xxt/SF-v3

N_SAMPLES=${N_SAMPLES:-20}
EPS=${EPS:-0.01}          # from pilot
SEED_BASE=${SEED_BASE:-100}
N_DIRS=${N_DIRS:-5}       # random perturbation directions per (sample, step)
PORT_BASE=${PORT_BASE:-29930}
OUTPUT_BASE=${OUTPUT_BASE:-results/exp4_amplification}
STEPS=(1 7 13 20 27 34 41 48)

DATASET_JSON=/public/home/maoyaoxin/xxt/datasets/exp4_${N_SAMPLES}samples.json
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python

if [ ! -f "${DATASET_JSON}" ]; then
  ${PY} <<EOF
import json, random
src = json.load(open("/public/home/maoyaoxin/xxt/datasets/sub-0005_test_va.json"))
random.seed(${SEED_BASE})
subset = random.sample(src, k=${N_SAMPLES})
json.dump(subset, open("${DATASET_JSON}", "w"))
print(f"Wrote {len(subset)} samples")
EOF
fi

mkdir -p ${OUTPUT_BASE}

# Baseline (no perturbation) shared across all steps and directions
BASELINE_DIR=${OUTPUT_BASE}/baseline
if [ ! -f "${BASELINE_DIR}/done.flag" ]; then
  mkdir -p ${BASELINE_DIR}
  CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/public/home/maoyaoxin/xxt/SF-v3 \
    nohup ${PY} -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=${PORT_BASE} \
    sample_brain_va.py --base \
      configs/sf_v1/cinebrain_sf_v1_model.yaml \
      configs/sf_v1/infer_stage3_v2.yaml \
    --seed 42 --jsonpath ${DATASET_JSON} \
    --output_dir ${BASELINE_DIR} \
    --save_latents \
    > ${OUTPUT_BASE}/baseline.log 2>&1
  touch ${BASELINE_DIR}/done.flag
fi

# Override YAML template for latent perturbation
cat > /tmp/exp4_latent_override_template.yaml <<'TEMPLATE'
model:
  sampler_config:
    params:
      perturb_spec:
        alpha_brain_step: null
        alpha_brain_delta: 0.0
        latent_step: LATENT_STEP
        latent_eps: LATENT_EPS
        latent_seed: LATENT_SEED
TEMPLATE

PORT=${PORT_BASE}
for idx in "${!STEPS[@]}"; do
  step=${STEPS[${idx}]}
  for d in $(seq 0 $((N_DIRS - 1))); do
    seed_val=$((SEED_BASE * 100 + step * 10 + d))
    out_dir=${OUTPUT_BASE}/step_${step}_dir_${d}
    if [ -f "${out_dir}/done.flag" ]; then
      continue
    fi
    override=/tmp/exp4_override_step${step}_dir${d}.yaml
    sed "s/LATENT_STEP/${step}/g; s/LATENT_EPS/${EPS}/g; s/LATENT_SEED/${seed_val}/g" \
      /tmp/exp4_latent_override_template.yaml > ${override}
    mkdir -p ${out_dir}

    # Alternate GPUs 0 and 1
    total_idx=$((idx * N_DIRS + d))
    gpu=$((total_idx % 2))
    PORT=$((PORT + 1))

    CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=${gpu} PYTHONPATH=/public/home/maoyaoxin/xxt/SF-v3 \
      nohup ${PY} -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=${PORT} \
      sample_brain_va.py --base \
        configs/sf_v1/cinebrain_sf_v1_model.yaml \
        ${override} \
        configs/sf_v1/infer_stage3_v2.yaml \
      --seed 42 --jsonpath ${DATASET_JSON} \
      --output_dir ${out_dir} \
      --save_latents \
      > ${OUTPUT_BASE}/step_${step}_dir_${d}.log 2>&1 &

    if [ $((total_idx % 2)) -eq 1 ]; then
      wait
      # Mark both as done
      prev_d=$((d - 1))
      if [ ${d} -eq 0 ]; then
        prev_step=${STEPS[$((idx - 1))]}
        prev_d=$((N_DIRS - 1))
        touch ${OUTPUT_BASE}/step_${prev_step}_dir_${prev_d}/done.flag
      else
        touch ${OUTPUT_BASE}/step_${step}_dir_${prev_d}/done.flag
      fi
      touch ${out_dir}/done.flag
    fi
  done
done
wait

for step in "${STEPS[@]}"; do
  for d in $(seq 0 $((N_DIRS - 1))); do
    touch ${OUTPUT_BASE}/step_${step}_dir_${d}/done.flag
  done
done

echo ""
echo "=== Exp 4 amplification sweep complete ==="
echo "Run analysis:"
echo "  ${PY} tools/analyze_ood_asymmetry.py --exp exp4 --base ${OUTPUT_BASE} --eps ${EPS}"
