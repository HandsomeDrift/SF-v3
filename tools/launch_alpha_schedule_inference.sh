#!/usr/bin/env bash
# Launch the 7 alpha-schedule inference runs sequentially on a single GPU.
# E0 is the v2 baseline (no schedule) — skip if you already have stage3_v2_sub05.
#
# Usage:
#   export CUDA_VISIBLE_DEVICES=0
#   bash tools/launch_alpha_schedule_inference.sh

set -euo pipefail

PROJ=/public/home/maoyaoxin/zhangt/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
MODEL=configs/sf_v1/cinebrain_sf_v1_model.yaml
V2_CKPT_CONFIG=configs/sf_v1/infer_stage3_v2.yaml
JSON=/public/home/maoyaoxin/zhangt/xxt/datasets/sub-0005_test_va.json

cd "$PROJ"
mkdir -p logs results

run_experiment () {
    local name=$1
    local schedule_yaml=$2   # "" for E0 baseline
    local outdir="results/v3_alpha_${name}"

    local bases=("$MODEL" "$V2_CKPT_CONFIG")
    [[ -n "$schedule_yaml" ]] && bases+=("$schedule_yaml")

    echo "[$(date '+%F %T')] launching $name → $outdir"
    CUDA_HOME=/usr/local/cuda-12.4 \
    nohup $PY -m torch.distributed.run --standalone --nproc_per_node=1 \
        sample_brain_va.py --base "${bases[@]}" \
        --seed 42 --jsonpath "$JSON" --output_dir "$outdir" \
        > "logs/alpha_${name}.log" 2>&1
    echo "[$(date '+%F %T')] done $name"
}

run_experiment E0_v2_static    ""
run_experiment E1_linear_mild   configs/sf_v1/alpha_schedule_E1_linear_mild.yaml
run_experiment E2_linear_strong configs/sf_v1/alpha_schedule_E2_linear_strong.yaml
run_experiment E3_cosine        configs/sf_v1/alpha_schedule_E3_cosine.yaml
run_experiment E4_sigmoid_mid   configs/sf_v1/alpha_schedule_E4_sigmoid_mid.yaml
run_experiment E5_sigmoid_early configs/sf_v1/alpha_schedule_E5_sigmoid_early.yaml
run_experiment E6_sigmoid_late  configs/sf_v1/alpha_schedule_E6_sigmoid_late.yaml

echo "All 7 runs finished. Aggregate with tools/eval_alpha_schedule.py."
