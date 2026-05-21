#!/bin/bash
# Probe iter 100/200/300/400/500 of sf_v3_pathB_cont500_lr1e6-04-24-16-06
set -e
cd /public/home/maoyaoxin/zhangt/xxt/SF-v3
export CUDA_HOME=/public/home/maoyaoxin/anaconda3/envs/cinebrain
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}
export PYTHONPATH=/public/home/maoyaoxin/zhangt/xxt/SF-v3:${PYTHONPATH}
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python

CKPT_DIR=ckpts_5b/sf_v3_pathB_cont500_lr1e6-04-24-16-06
ORIG_LATEST=$(cat $CKPT_DIR/latest)
trap "echo $ORIG_LATEST > $CKPT_DIR/latest; echo restored latest to $ORIG_LATEST" EXIT

mkdir -p results/pathB

for STEP in 100 200 300 400 500; do
  echo "========== PROBE cont500 iter $STEP =========="
  date
  echo $STEP > $CKPT_DIR/latest
  $PY -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=29978 \
    tools/probe_pathB_alpha_curve.py \
    --base configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml \
           configs/sf_v1/infer_stage3_v2.yaml \
    --ckpt_path $CKPT_DIR \
    --jsonpath /public/home/maoyaoxin/zhangt/xxt/datasets/sub-0005_test_va.json \
    --max_samples 50 \
    --dump_output results/pathB/alpha_curve_cont500_iter${STEP}.json \
    --seed 42
done
echo "========== ALL DONE =========="
date
