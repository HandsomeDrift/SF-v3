#!/usr/bin/env bash
# Parallel 4+4 eval: iter 0 on gpu5 GPU 4-7, iter 500 on gpu5 GPU 0-3.
# 4-way split → 135 samples/GPU × ~4.5 min ≈ 10h wall clock each (both finish together).
set -u
PROJ=/public/home/maoyaoxin/zhangt/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
MODEL=configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml
cd $PROJ
mkdir -p logs
: > logs/pathB_iter0_500_4way.pids

export CUDA_HOME=/public/home/maoyaoxin/anaconda3/envs/cinebrain
export PYTHONPATH=${PROJ}:${PYTHONPATH:-}

launch_split() {
    local iter=$1; local gpu=$2; local split=$3; local port=$4
    local infer=configs/sf_v1/infer_pathB_p1_iter${iter}.yaml
    local outdir=results/alpha_540/pathB_p1_iter${iter}
    local jsonpath=/public/home/maoyaoxin/zhangt/xxt/datasets/full540_4split${split}.json
    local log=logs/pathB_iter${iter}_gpu5_gpu${gpu}.log
    mkdir -p ${outdir}
    CUDA_VISIBLE_DEVICES=${gpu} \
      nohup ${PY} -m torch.distributed.run \
          --standalone --nproc_per_node=1 --master_port=${port} \
          sample_brain_va.py \
          --base ${MODEL} ${infer} \
          --seed 42 --jsonpath ${jsonpath} --output_dir ${outdir} \
          > ${log} 2>&1 &
    echo "iter${iter} gpu5 gpu${gpu} split${split} pid=$!" >> logs/pathB_iter0_500_4way.pids
}

# iter 500 on GPU 0-3 (fresh start — existing dir may have 56 partial, clean first)
rm -rf results/alpha_540/pathB_p1_iter500
for i in 0 1 2 3; do launch_split 500 $i $i $((29850 + i)); done

# iter 0 on GPU 4-7
rm -rf results/alpha_540/pathB_p1_iter0
for i in 0 1 2 3; do launch_split 0 $((i + 4)) $i $((29860 + i)); done

echo "=== launched 4+4 processes ==="
cat logs/pathB_iter0_500_4way.pids
