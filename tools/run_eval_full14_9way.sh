#!/usr/bin/env bash
# Run full 14-metric eval on all 9 Path A/B full-540 configs.
# Single GPU, sequential, ~15 min × 9 = ~2-2.5h total.
set -euo pipefail

PROJ=/public/home/maoyaoxin/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
EVAL_GPU=${EVAL_GPU:-0}
GT_JSON=/public/home/maoyaoxin/xxt/datasets/sub-0005_test_va.json
RESULTS=${PROJ}/results/alpha_540
OUT=${RESULTS}/summary_9way_pathB_full14.json
cd ${PROJ}

export CUDA_HOME=/public/home/maoyaoxin/anaconda3/envs/cinebrain
export PYTHONPATH=${PROJ}
export CUDA_VISIBLE_DEVICES=${EVAL_GPU}

${PY} tools/eval_full14.py \
    --gt-jsonpath ${GT_JSON} \
    --result-dir "E0_new_code=${RESULTS}/E0_new_code" \
    --result-dir "E3_cosine=${RESULTS}/E3_cosine" \
    --result-dir "E4_reverse=${RESULTS}/E4_reverse" \
    --result-dir "E4_reverse_clamped=${RESULTS}/E4_reverse_clamped" \
    --result-dir "E4_sigmoid_mid=${RESULTS}/E4_sigmoid_mid" \
    --result-dir "E4_sigmoid_mid_clamped=${RESULTS}/E4_sigmoid_mid_clamped" \
    --result-dir "pathB_p1_iter2000=${RESULTS}/pathB_p1_iter2000" \
    --result-dir "pathB_p1_iter1500=${RESULTS}/pathB_p1_iter1500" \
    --result-dir "pathB_p1_iter1000=${RESULTS}/pathB_p1_iter1000" \
    --baseline E4_reverse \
    --output ${OUT}
