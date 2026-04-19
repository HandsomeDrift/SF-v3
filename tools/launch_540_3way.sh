#!/usr/bin/env bash
# 540-sample full inference for 3 experiments across 9 GPUs on 2 nodes.
# Split: 540 / 3 cards = 180 samples per card × 4.5 min ≈ 13.5h wall clock.
#
# Node/GPU map:
#   gpu2 GPU 0/1/2 → E3_cosine           (amp=0.4)
#   gpu2 GPU 3/6/7 → E4_sigmoid_mid      (amp=0.5)
#   gpu1 GPU 0/1/2 → E4_reverse          (amp=-0.5)
#
# GPU 4/5 on gpu2 are root's vllm (do not touch).
#
# Usage (run from ts3 login node):
#   bash /public/home/maoyaoxin/xxt/SF-v3/tools/launch_540_3way.sh

set -u
PROJ=/public/home/maoyaoxin/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
MODEL=configs/sf_v1/cinebrain_sf_v1_model.yaml
V2_CKPT=configs/sf_v1/infer_stage3_v2.yaml

# Wipe & recreate output dirs / pid log
ssh gpu2 "cd $PROJ && rm -rf results/alpha_540/E3_cosine results/alpha_540/E4_sigmoid_mid && \
    mkdir -p results/alpha_540/E3_cosine results/alpha_540/E4_sigmoid_mid logs && \
    : > logs/alpha_540_3way.pids"
ssh gpu1 "cd $PROJ && rm -rf results/alpha_540/E4_reverse && \
    mkdir -p results/alpha_540/E4_reverse logs"

launch_on () {
    local host=$1 gpu=$2 name=$3 schedule_yaml=$4 split=$5
    local outdir="results/alpha_540/${name}"
    local log="logs/alpha_540_${name}_${host}_gpu${gpu}.log"
    local port=$((29900 + gpu))
    local jsonpath="/public/home/maoyaoxin/xxt/datasets/full540_split${split}.json"

    # Trailing '&' backgrounds the ssh call itself so all 9 launches fire in parallel.
    ssh "$host" "cd $PROJ && CUDA_HOME=/usr/local/cuda-12.4 CUDA_VISIBLE_DEVICES=${gpu} \
        nohup ${PY} -m torch.distributed.run \
            --standalone --nproc_per_node=1 --master_port=${port} \
            sample_brain_va.py \
            --base ${MODEL} ${V2_CKPT} ${schedule_yaml} \
            --seed 42 --jsonpath ${jsonpath} --output_dir ${outdir} \
            > ${log} 2>&1 & echo \"${name} ${host} gpu${gpu} split${split} pid=\$!\" >> logs/alpha_540_3way.pids" &
}

# E3_cosine on gpu2 0/1/2
launch_on gpu2 0 E3_cosine      configs/sf_v1/alpha_schedule_E3_cosine.yaml       0
launch_on gpu2 1 E3_cosine      configs/sf_v1/alpha_schedule_E3_cosine.yaml       1
launch_on gpu2 2 E3_cosine      configs/sf_v1/alpha_schedule_E3_cosine.yaml       2

# E4_sigmoid_mid on gpu2 3/6/7
launch_on gpu2 3 E4_sigmoid_mid configs/sf_v1/alpha_schedule_E4_sigmoid_mid.yaml  0
launch_on gpu2 6 E4_sigmoid_mid configs/sf_v1/alpha_schedule_E4_sigmoid_mid.yaml  1
launch_on gpu2 7 E4_sigmoid_mid configs/sf_v1/alpha_schedule_E4_sigmoid_mid.yaml  2

# E4_reverse on gpu1 0/1/2
launch_on gpu1 0 E4_reverse     configs/sf_v1/alpha_schedule_E4_reverse.yaml       0
launch_on gpu1 1 E4_reverse     configs/sf_v1/alpha_schedule_E4_reverse.yaml       1
launch_on gpu1 2 E4_reverse     configs/sf_v1/alpha_schedule_E4_reverse.yaml       2

# Wait for all 9 ssh calls to complete (each is fast: starts nohup + exits)
wait

echo "=== launched 9 processes ==="
ssh gpu2 "cat $PROJ/logs/alpha_540_3way.pids"
