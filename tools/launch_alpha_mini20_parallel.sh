#!/usr/bin/env bash
# Launch 4 alpha-schedule inference runs in parallel on gpu2 GPUs 0-3 using
# the 20-sample mini-test subset. Each run pins itself to one GPU and writes
# to a dedicated output_dir/log. Caller is expected to wait for all four
# background jobs to finish before aggregating metrics.
#
# Usage:
#   bash tools/launch_alpha_mini20_parallel.sh <experiment_index_set>
# Where <experiment_index_set> is one of:
#   round1   → E3,E4,E5,E6   (GPUs 0,1,2,3)
#   round2   → E1,E2         (GPUs 0,1)
# Or pass a custom list like: "E3_cosine:0 E5_sigmoid_early:1"

set -euo pipefail

PROJ=/public/home/maoyaoxin/zhangt/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
MODEL=configs/sf_v1/cinebrain_sf_v1_model.yaml
V2_CKPT=configs/sf_v1/infer_stage3_v2.yaml
JSON=/public/home/maoyaoxin/zhangt/xxt/datasets/sub-0005_test_va_mini20.json

cd "$PROJ"
mkdir -p logs "results/alpha_mini20"

round="${1:-round1}"
case "$round" in
    round1)
        SPECS=("E3_cosine:0" "E4_sigmoid_mid:1" "E5_sigmoid_early:2" "E6_sigmoid_late:3")
        ;;
    round2)
        SPECS=("E1_linear_mild:0" "E2_linear_strong:1")
        ;;
    *)
        # Custom space-separated list from command line
        IFS=' ' read -ra SPECS <<< "$round"
        ;;
esac

pids=()
for spec in "${SPECS[@]}"; do
    name="${spec%:*}"
    gpu="${spec#*:}"
    outdir="results/alpha_mini20/${name}"
    log="logs/alpha_mini20_${name}.log"
    port=$((29500 + gpu))
    schedule_yaml="configs/sf_v1/alpha_schedule_${name}.yaml"

    echo "[launch] name=$name gpu=$gpu port=$port out=$outdir"
    mkdir -p "$outdir"
    CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES="$gpu" \
    nohup $PY -m torch.distributed.run \
        --standalone --nproc_per_node=1 --master_port="$port" \
        sample_brain_va.py \
        --base "$MODEL" "$V2_CKPT" "$schedule_yaml" \
        --seed 42 --jsonpath "$JSON" --output_dir "$outdir" \
        > "$log" 2>&1 &
    pids+=("$!")
done

echo "[launched] pids=${pids[*]}"
echo "[waiting]  $(date '+%F %T')"
wait "${pids[@]}"
echo "[done]     $(date '+%F %T')"
