#!/usr/bin/env bash
# Eval Path B iter 1500 540-sample metrics, then aggregate into 8-way table.
# Run AFTER inference completes (540/540 mp4s in results/alpha_540/pathB_p1_iter1000/).
# Single GPU, ~10 min.
set -euo pipefail

PROJ=/public/home/maoyaoxin/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
EVAL_GPU=${EVAL_GPU:-0}
GT_JSON=/public/home/maoyaoxin/xxt/datasets/sub-0005_test_va.json
RESULTS=${PROJ}/results/alpha_540
OUT_DIR=${RESULTS}/pathB_p1_iter1000
SUMMARY_1500=${RESULTS}/summary_pathB_p1_iter1000.json
SUMMARY_8WAY=${RESULTS}/summary_9way_pathB.json

cd ${PROJ}

# pre-flight
count=$(ls "${OUT_DIR}"/*.mp4 2>/dev/null | wc -l)
echo "=== pathB_p1_iter1000: ${count}/540 mp4s ==="
if [ "${count}" -lt 540 ]; then
  echo "WARNING: only ${count}/540, proceeding anyway"
fi

# (1) single-model eval
echo ""
echo "=== eval pathB_p1_iter1000 on GPU ${EVAL_GPU} ==="
CUDA_HOME=/public/home/maoyaoxin/anaconda3/envs/cinebrain \
CUDA_VISIBLE_DEVICES=${EVAL_GPU} \
PYTHONPATH=${PROJ} ${PY} tools/eval_alpha_schedule.py \
  --gt-jsonpath ${GT_JSON} \
  --result-dir "pathB_p1_iter1000=${OUT_DIR}" \
  --baseline "pathB_p1_iter1000" \
  --output ${SUMMARY_1500}

echo ""
echo "=== aggregate 8-way (7-way + iter1000) ==="
${PY} - << PYEOF
import json
base = json.load(open("${RESULTS}/summary_7way_pathB.json"))
new = json.load(open("${SUMMARY_1500}"))
key = "pathB_p1_iter1000"
row = new.get(key) or new["experiments"][key]
base["experiments"][key] = row
json.dump(base, open("${SUMMARY_8WAY}", "w"), indent=2)
print(f"wrote ${SUMMARY_8WAY} with {len(base[\"experiments\"])} entries")

# Print markdown table with Delta vs E4_reverse and iter 2000
rows = [("E0_new_code","baseline"),("E3_cosine","Path A cosine"),("E4_reverse","Path A winner"),
        ("E4_reverse_clamped","H** anchor"),("E4_sigmoid_mid","Path A sigmoid"),
        ("E4_sigmoid_mid_clamped","H** clamp"),("pathB_p1_iter2000","Path B overtrained"),
        ("pathB_p1_iter1000","Path B sweet spot (NEW)")]
print("\n| Exp | FVD | EPE | SSIM | CLIP |")
print("|---|---:|---:|---:|---:|")
e = base["experiments"]
for k, tag in rows:
    m = e.get(k)
    if not m: continue
    print(f"| {k} ({tag}) | {m[FVD]:.1f} | {m[EPE]:.2f} | {m[SSIM]:.3f} | {m[CLIP]:.3f} |")
# diff vs iter 2000
b1500 = e.get("pathB_p1_iter1000"); b2000 = e.get("pathB_p1_iter2000")
if b1500 and b2000:
    d = b1500["FVD"] - b2000["FVD"]
    print(f"\niter 1500 vs iter 2000: FVD Delta = {d:+.1f} ({d/b2000[FVD]*100:+.1f}%)")
# diff vs E4_reverse
b_rev = e.get("E4_reverse")
if b1500 and b_rev:
    d = b1500["FVD"] - b_rev["FVD"]
    print(f"iter 1500 vs E4_reverse: FVD Delta = {d:+.1f} ({d/b_rev[FVD]*100:+.1f}%)")
PYEOF

echo ""
echo "=== DONE ==="
ls -la ${SUMMARY_1500} ${SUMMARY_8WAY}
