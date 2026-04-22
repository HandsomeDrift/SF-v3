#!/usr/bin/env bash
# Eval 4 α-clamp configs (none/mot/brain/both) on mini-68.
# Runs sequentially on 1 GPU, ~10 min × 4 = ~40 min total.
set -euo pipefail
PROJ=/public/home/maoyaoxin/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
EVAL_GPU=${EVAL_GPU:-0}
GT_JSON=/public/home/maoyaoxin/xxt/datasets/full540_8split0.json
RESULTS=${PROJ}/results/alpha_540
cd ${PROJ}

for n in none mot brain both; do
  DIR=${RESULTS}/pathB_iter1500_clamp_${n}_mini68
  OUT=${RESULTS}/summary_pathB_iter1500_clamp_${n}_mini68.json
  count=$(ls ${DIR}/*.mp4 2>/dev/null | wc -l)
  echo "=== clamp=${n}: ${count}/68 ==="
  if [ "${count}" -lt 60 ]; then
    echo "SKIP (${count}<60)"; continue
  fi
  CUDA_HOME=/public/home/maoyaoxin/anaconda3/envs/cinebrain \
  CUDA_VISIBLE_DEVICES=${EVAL_GPU} \
  PYTHONPATH=${PROJ} ${PY} tools/eval_alpha_schedule.py \
    --gt-jsonpath ${GT_JSON} \
    --result-dir "pathB_clamp_${n}_mini68=${DIR}" \
    --baseline "pathB_clamp_${n}_mini68" \
    --output ${OUT}
done

echo ""
echo "=== aggregate 4-way clamp table ==="
${PY} /tmp/agg_clamp.py
