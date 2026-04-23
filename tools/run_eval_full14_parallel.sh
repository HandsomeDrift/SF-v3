#!/usr/bin/env bash
# Parallel 14-metric eval: 2 workers only (CPU RAM constraint).
# 2 × ~41 = 82 GB peak, safe on 251 GB node. Wall clock ~100 min.
# Important configs first in each chain so partial crash preserves key data.
set -u
PROJ=/public/home/maoyaoxin/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
GT_JSON=/public/home/maoyaoxin/xxt/datasets/sub-0005_test_va.json
R=${PROJ}/results/alpha_540
cd ${PROJ}
export CUDA_HOME=/public/home/maoyaoxin/anaconda3/envs/cinebrain
export PYTHONPATH=${PROJ}
mkdir -p logs
: > logs/eval_full14_parallel.pids

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

# Chain A (GPU 0): 5 configs — Path B trio + Path A winner + baseline (priority-first)
(run_one 0 pathB_p1_iter1000 \
 && run_one 0 pathB_p1_iter1500 \
 && run_one 0 pathB_p1_iter2000 \
 && run_one 0 E4_reverse \
 && run_one 0 E0_new_code) &
echo "gpu0 chain5 pid=$!" >> logs/eval_full14_parallel.pids

# Chain B (GPU 1): 4 configs — clamp + sigmoid pair + cosine
(run_one 1 E4_reverse_clamped \
 && run_one 1 E4_sigmoid_mid \
 && run_one 1 E4_sigmoid_mid_clamped \
 && run_one 1 E3_cosine) &
echo "gpu1 chain4 pid=$!" >> logs/eval_full14_parallel.pids

echo "=== launched 2 chains (9 configs total) ==="
cat logs/eval_full14_parallel.pids
