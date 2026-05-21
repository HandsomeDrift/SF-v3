#!/bin/bash
# Probe iter 500 / 1000 / 1500 ckpts. Uses deepspeed-style latest-file loader:
# temporarily overwrite {ckpt_dir}/latest to force each iter, restore at end.
set -e
cd /public/home/maoyaoxin/zhangt/xxt/SF-v3
export CUDA_HOME=/public/home/maoyaoxin/anaconda3/envs/cinebrain
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH=/public/home/maoyaoxin/zhangt/xxt/SF-v3:${PYTHONPATH}
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python

CKPT_DIR=ckpts_5b/sf_v3_pathB_p1-04-19-01-13
ORIG_LATEST=$(cat $CKPT_DIR/latest)
trap "echo $ORIG_LATEST > $CKPT_DIR/latest; echo restored latest to $ORIG_LATEST" EXIT

for STEP in 500 1000 1500; do
  echo "========== PROBE iter $STEP =========="
  date
  echo $STEP > $CKPT_DIR/latest
  $PY -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=29977 \
    tools/probe_pathB_alpha_curve.py \
    --base configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml \
           configs/sf_v1/infer_stage3_v2.yaml \
    --ckpt_path $CKPT_DIR \
    --jsonpath /public/home/maoyaoxin/zhangt/xxt/datasets/sub-0005_test_va.json \
    --max_samples 50 \
    --dump_output results/pathB/alpha_curve_p1_iter${STEP}.json \
    --seed 42
done
echo "========== ALL DONE =========="
date
