#!/usr/bin/env bash
# Parallel eval of 3 missing configs that failed on gpu5 (E4_reverse chain).
# Designed to run on a fresh node with >200 GB RAM free (gpu7 verified).
# Runs 3 configs in parallel on 3 GPUs, ~20 min wall clock.
set -u
PROJ=/public/home/maoyaoxin/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
GT_JSON=/public/home/maoyaoxin/xxt/datasets/sub-0005_test_va.json
R=${PROJ}/results/alpha_540
cd ${PROJ}
export CUDA_HOME=/public/home/maoyaoxin/anaconda3/envs/cinebrain
export PYTHONPATH=${PROJ}
mkdir -p logs
: > logs/eval_full14_missing3.pids

run_one() {
    local gpu=$1; local name=$2
    local out=${R}/summary_${name}_full14.json
    local log=logs/eval_full14_${name}.log
    : > ${log}
    CUDA_VISIBLE_DEVICES=${gpu} ${PY} tools/eval_full14.py \
        --gt-jsonpath ${GT_JSON} \
        --result-dir "${name}=${R}/${name}" \
        --output ${out} \
        > ${log} 2>&1
}

# 3 configs, 3 GPUs (parallel) — on gpu4 GPU 0, 1, 2
(run_one 0 E4_reverse) & echo "E4_reverse pid=$!" >> logs/eval_full14_missing3.pids
(run_one 1 E4_reverse_clamped) & echo "E4_reverse_clamped pid=$!" >> logs/eval_full14_missing3.pids
(run_one 2 E4_sigmoid_mid) & echo "E4_sigmoid_mid pid=$!" >> logs/eval_full14_missing3.pids

echo "=== launched 3 parallel ==="
cat logs/eval_full14_missing3.pids
