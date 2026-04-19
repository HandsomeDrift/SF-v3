#!/usr/bin/env bash
# E4_sigmoid_mid_clamped 540-sample inference, 8-way split on gpu2.
# H** validation: does alpha_max=0.95 clamp rescue FVD from 1194 back to ~700-800?
#
# Run on gpu2 directly (dispatched from ts3 via `ssh ts3 "ssh gpu2 bash <this>"`).
# Wall clock: 540 / 8 × 4.5 min ≈ 5.1h.

set -u
PROJ=/public/home/maoyaoxin/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
MODEL=configs/sf_v1/cinebrain_sf_v1_model.yaml
V2_CKPT=configs/sf_v1/infer_stage3_v2.yaml
SCHEDULE=configs/sf_v1/alpha_schedule_E4_clamped.yaml
NAME=E4_sigmoid_mid_clamped
OUTDIR=results/alpha_540/${NAME}

cd $PROJ
rm -rf ${OUTDIR}
mkdir -p ${OUTDIR} logs
: > logs/alpha_540_clamp_8way.pids

for gpu in 0 1 2 3 4 5 6 7; do
    split=${gpu}
    log=logs/alpha_540_${NAME}_gpu2_gpu${gpu}.log
    port=$((29900 + gpu))
    jsonpath=/public/home/maoyaoxin/xxt/datasets/full540_8split${split}.json

    CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=${gpu} \
      nohup ${PY} -m torch.distributed.run \
          --standalone --nproc_per_node=1 --master_port=${port} \
          sample_brain_va.py \
          --base ${MODEL} ${V2_CKPT} ${SCHEDULE} \
          --seed 42 --jsonpath ${jsonpath} --output_dir ${OUTDIR} \
          > ${log} 2>&1 &
    echo "${NAME} gpu2 gpu${gpu} split${split} pid=$!" >> logs/alpha_540_clamp_8way.pids
done

echo "=== launched 8 processes ==="
cat logs/alpha_540_clamp_8way.pids
