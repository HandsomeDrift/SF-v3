#!/usr/bin/env bash
# Quick 1-sample 2-config α dump to verify clamp actually changes α values.
# Runs clamp=none and clamp=both in parallel on gpu5 GPU 0 and 1.
# Each ~5 min.
set -u
PROJ=/public/home/maoyaoxin/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
MODEL=configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml
INFER=configs/sf_v1/infer_pathB_p1_iter1500.yaml
SPLIT=/public/home/maoyaoxin/xxt/datasets/single_sample.json
cd $PROJ
export CUDA_HOME=/public/home/maoyaoxin/anaconda3/envs/cinebrain
export PYTHONPATH=${PROJ}:${PYTHONPATH:-}
export DBG_ALPHA_STEP=25
mkdir -p logs tmp_dbg

# none
CUDA_VISIBLE_DEVICES=0 \
  nohup ${PY} -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=29830 \
      sample_brain_va.py --base ${MODEL} ${INFER} \
      --seed 42 --jsonpath ${SPLIT} --output_dir tmp_dbg/none_1s \
      > logs/dbg_alpha_none.log 2>&1 &
echo "none pid=$!"

# both
CUDA_VISIBLE_DEVICES=1 \
  nohup ${PY} -m torch.distributed.run --standalone --nproc_per_node=1 --master_port=29831 \
      sample_brain_va.py --base ${MODEL} ${INFER} configs/sf_v1/pathB_clamp_both.yaml \
      --seed 42 --jsonpath ${SPLIT} --output_dir tmp_dbg/both_1s \
      > logs/dbg_alpha_both.log 2>&1 &
echo "both pid=$!"
