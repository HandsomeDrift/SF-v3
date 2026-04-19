#!/bin/bash
# §6 Exp 1 PILOT - Single-GPU sequential version.
# Runs on GPU 1 only (GPU 0 occupied by another user as of 2026-04-20 01:10).
# N_SAMPLES=5 × 8 steps + 1 baseline = 45 runs × ~7min = ~5.25 hours wall-clock.
#
# FORCE_DETERMINISM=1 is essential - unit test showed without it, GPU non-det
# creates 0.17/element noise floor that swamps perturbation signal.
# With it, T1 bit-identical, T4 monotonicity ratio 98x on 1 sample.

set -eo pipefail

cd /public/home/maoyaoxin/xxt/SF-v3

N_SAMPLES=${N_SAMPLES:-5}
PERTURB_DELTA=${PERTURB_DELTA:-0.3}
GPU=${GPU:-1}
PORT_BASE=${PORT_BASE:-29920}
OUTPUT_BASE=${OUTPUT_BASE:-results/exp1_pilot_n${N_SAMPLES}}
STEPS=(1 7 13 20 27 34 41 48)

DATASET_JSON=/public/home/maoyaoxin/xxt/datasets/exp1_pilot_${N_SAMPLES}samples.json
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python

export CUDA_HOME=/usr/local/cuda-12.4
export CUDA_VISIBLE_DEVICES=${GPU}
export PYTHONPATH=/public/home/maoyaoxin/xxt/SF-v3
export FORCE_DETERMINISM=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [ ! -f "${DATASET_JSON}" ]; then
  echo "Generating ${N_SAMPLES}-sample dataset JSON..."
  ${PY} <<EOF
import json, random
src = json.load(open("/public/home/maoyaoxin/xxt/datasets/sub-0005_test_va.json"))
random.seed(42)
subset = random.sample(src, k=${N_SAMPLES})
json.dump(subset, open("${DATASET_JSON}", "w"))
print(f"Wrote {len(subset)} samples to ${DATASET_JSON}")
EOF
fi

mkdir -p ${OUTPUT_BASE}

run_one() {
  local tag=$1
  local override=$2
  local port=$3
  local out_dir=${OUTPUT_BASE}/${tag}

  if [ -f "${out_dir}/done.flag" ] && [ "$(ls ${out_dir}/*_latent.pt 2>/dev/null | wc -l)" -eq "${N_SAMPLES}" ]; then
    echo "=== ${tag} already complete, skipping ==="
    return
  fi

  echo "=== [${tag}] GPU ${GPU} port ${port}  override=${override:-<none>} ==="
  mkdir -p ${out_dir}

  local base_args=("configs/sf_v1/cinebrain_sf_v1_model.yaml")
  [ -n "${override}" ] && base_args+=("${override}")
  base_args+=("configs/sf_v1/infer_stage3_v2.yaml")

  local t0=$(date +%s)
  ${PY} -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=${port} \
    sample_brain_va.py --base "${base_args[@]}" \
    --seed 42 \
    --jsonpath ${DATASET_JSON} \
    --output_dir ${out_dir} \
    --save_latents \
    > ${OUTPUT_BASE}/${tag}.log 2>&1
  local rc=$?
  local t1=$(date +%s)
  local elapsed=$((t1 - t0))
  local n_lat=$(ls ${out_dir}/*_latent.pt 2>/dev/null | wc -l)
  echo "    Done in ${elapsed}s, ${n_lat} latent(s) saved (rc=${rc})"
  if [ "${n_lat}" -eq "${N_SAMPLES}" ]; then
    touch ${out_dir}/done.flag
  else
    echo "    WARNING: expected ${N_SAMPLES} latents, got ${n_lat}"
  fi
}

# Baseline (no perturbation, shared across all step comparisons)
run_one "baseline" "" ${PORT_BASE}

# Step sweep
PORT=${PORT_BASE}
for step in "${STEPS[@]}"; do
  PORT=$((PORT + 1))
  override=/tmp/exp1_pilot_override_step${step}.yaml
  sed "s/PERTURB_STEP/${step}/g; s/PERTURB_DELTA/${PERTURB_DELTA}/g" \
    configs/sf_v1/exp1_perturb_override.yaml > ${override}
  run_one "step_${step}" "${override}" ${PORT}
done

echo ""
echo "=========================================="
echo "  Exp 1 pilot complete (N=${N_SAMPLES})"
echo "=========================================="
echo "Run analysis:"
echo "  ${PY} tools/analyze_ood_asymmetry.py --exp exp1 --base ${OUTPUT_BASE} --plot"
