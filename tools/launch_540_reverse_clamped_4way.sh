#!/usr/bin/env bash
# E4_reverse_clamped 540-sample inference, 4-way split on gpu1 GPU 0-3.
# H** secondary validation: does alpha_max=0.95 clamp on reverse schedule
# further improve FVD beyond 425.28, revealing late-step OOD contribution?
#
# Dispatch from ts3 login via `ssh ts3 "ssh gpu1 bash <this>"`.
# Wall clock: 540 / 4 × 4.5 min ≈ 10.1h.

set -u
PROJ=/public/home/maoyaoxin/zhangt/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
MODEL=configs/sf_v1/cinebrain_sf_v1_model.yaml
V2_CKPT=configs/sf_v1/infer_stage3_v2.yaml
SCHEDULE=configs/sf_v1/alpha_schedule_E4_reverse_clamped.yaml
NAME=E4_reverse_clamped
OUTDIR=results/alpha_540/${NAME}

cd $PROJ
rm -rf ${OUTDIR}
mkdir -p ${OUTDIR} logs
: > logs/alpha_540_reverse_clamped.pids

for gpu in 0 1 2 3; do
    split=${gpu}
    log=logs/alpha_540_${NAME}_gpu1_gpu${gpu}.log
    port=$((29900 + gpu))
    jsonpath=/public/home/maoyaoxin/zhangt/xxt/datasets/full540_4split${split}.json

    CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=${gpu} \
      nohup ${PY} -m torch.distributed.run \
          --standalone --nproc_per_node=1 --master_port=${port} \
          sample_brain_va.py \
          --base ${MODEL} ${V2_CKPT} ${SCHEDULE} \
          --seed 42 --jsonpath ${jsonpath} --output_dir ${OUTDIR} \
          > ${log} 2>&1 &
    echo "${NAME} gpu1 gpu${gpu} split${split} pid=$!" >> logs/alpha_540_reverse_clamped.pids
done

echo "=== launched 4 processes ==="
cat logs/alpha_540_reverse_clamped.pids
