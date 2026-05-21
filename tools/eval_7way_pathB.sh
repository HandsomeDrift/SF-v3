#!/usr/bin/env bash
# 7-way evaluation: Path B P1 iter2000 vs. 6 static-schedule baselines.
#
# NB: 6 baselines already have per-experiment summary JSON under
# results/alpha_540/summary_*.json. This script only needs to:
#   (1) eval pathB_p1_iter2000 (~10 min on one H800)
#   (2) aggregate all 7 summaries into a comparison table
#
# Dispatch from ts3 login via `ssh ts3 "ssh gpu2 bash <this>"` or run directly
# on gpu2 after the Path B 540 inference completes.
#
# Output:
#   - results/alpha_540/summary_pathB_p1_iter2000.json (new)
#   - results/alpha_540/summary_7way_pathB.json        (aggregated)
#   - stdout: 7-way markdown table with Δ vs E0_new_code

set -euo pipefail

PROJ=/public/home/maoyaoxin/zhangt/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
EVAL_GPU=${EVAL_GPU:-0}
GT_JSON=/public/home/maoyaoxin/zhangt/xxt/datasets/sub-0005_test_va.json
RESULTS=${PROJ}/results/alpha_540
PATHB_DIR=${RESULTS}/pathB_p1_iter2000
PATHB_SUMMARY=${RESULTS}/summary_pathB_p1_iter2000.json

cd ${PROJ}

# --- pre-flight: check Path B inference completeness ---
echo "=== pre-flight: pathB_p1_iter2000 sample count ==="
if [[ ! -d "${PATHB_DIR}" ]]; then
    echo "FATAL: ${PATHB_DIR} does not exist"
    exit 1
fi
count=$(ls "${PATHB_DIR}"/*.mp4 2>/dev/null | wc -l)
echo "  pathB_p1_iter2000: ${count}/540 mp4s"
if [[ ${count} -lt 540 ]]; then
    echo ""
    echo "WARNING: inference not complete (${count}/540). Abort? [y/N]"
    read -t 10 -r ans || ans="y"
    if [[ "${ans}" =~ ^[yY]$ ]]; then
        echo "aborted."
        exit 1
    fi
    echo "continuing despite incompleteness..."
fi
echo ""

# --- (1) eval only pathB_p1_iter2000 (re-using cached GT load + metric models) ---
echo "=== eval pathB_p1_iter2000 on GPU ${EVAL_GPU} ==="
CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=${EVAL_GPU} \
    PYTHONPATH=${PROJ} ${PY} tools/eval_alpha_schedule.py \
    --gt-jsonpath ${GT_JSON} \
    --result-dir "pathB_p1_iter2000=${PATHB_DIR}" \
    --baseline "pathB_p1_iter2000" \
    --output ${PATHB_SUMMARY}

echo ""

# --- (2) aggregate 7-way comparison table ---
echo "=== aggregate 7-way summary ==="
PYTHONPATH=${PROJ} ${PY} tools/aggregate_7way_summary.py \
    --summary-root ${RESULTS} \
    --baseline E0_new_code \
    --output ${RESULTS}/summary_7way_pathB.json

echo ""
echo "=== done ==="
echo ""
echo "Decision tree (based on pathB_p1_iter2000 FVD):"
echo "  FVD < 400                → Strong → Direction ② probe next"
echo "  400 <= FVD < 450         → Mild → probe + P2 TimeNoise preparation"
echo "  FVD ≈ 425 (≈ E4_reverse) → Match → diagnose what gate_net learned"
echo "  450 <= FVD < 500         → Degrade → iter 500/1000/1500 trajectory probe"
echo "  FVD > 500                → Null → Path B has training issues, diagnose"
