#!/usr/bin/env bash
# α-clamp ablation on Path B iter 1500:
#   4 configs (none/mot/brain/both) × 2 GPUs each = 8 GPUs on gpu5
#   mini-68 split 2-way → 34 samples/GPU, ~34 × 4.5 min ≈ 2.55h wall clock.
# Purpose: decompose which gate_net drift channel controls EPE vs FVD.
set -u
PROJ=/public/home/maoyaoxin/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
MODEL=configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml
INFER=configs/sf_v1/infer_pathB_p1_iter1500.yaml
SPLIT_A=/public/home/maoyaoxin/xxt/datasets/mini68_halfA.json
SPLIT_B=/public/home/maoyaoxin/xxt/datasets/mini68_halfB.json
DSET_DIR=results/alpha_540

cd $PROJ
mkdir -p logs

export CUDA_HOME=/public/home/maoyaoxin/anaconda3/envs/cinebrain
export PYTHONPATH=${PROJ}:${PYTHONPATH:-}

: > logs/pathB_clamp_mini68.pids

# 4 configs: name -> override yaml (empty = no override, pure iter1500)
declare -A CFG=(
    [none]=""
    [mot]="configs/sf_v1/pathB_clamp_mot.yaml"
    [brain]="configs/sf_v1/pathB_clamp_brain.yaml"
    [both]="configs/sf_v1/pathB_clamp_both.yaml"
)
# GPU assignment: 2 GPUs per config
declare -A GPU_A=([none]=0 [mot]=2 [brain]=4 [both]=6)
declare -A GPU_B=([none]=1 [mot]=3 [brain]=5 [both]=7)
declare -A PORT_A=([none]=29810 [mot]=29812 [brain]=29814 [both]=29816)
declare -A PORT_B=([none]=29811 [mot]=29813 [brain]=29815 [both]=29817)

for name in none mot brain both; do
    OUT=${DSET_DIR}/pathB_iter1500_clamp_${name}_mini68
    rm -rf ${OUT}
    mkdir -p ${OUT}
    override=${CFG[$name]}

    for half in A B; do
        if [ "$half" = "A" ]; then
            gpu=${GPU_A[$name]}; port=${PORT_A[$name]}; split=${SPLIT_A}
        else
            gpu=${GPU_B[$name]}; port=${PORT_B[$name]}; split=${SPLIT_B}
        fi
        log=logs/pathB_clamp_${name}_half${half}_gpu${gpu}.log

        BASE_ARGS="${MODEL} ${INFER}"
        [ -n "$override" ] && BASE_ARGS="${BASE_ARGS} ${override}"

        CUDA_VISIBLE_DEVICES=${gpu} \
          nohup ${PY} -m torch.distributed.run \
              --standalone --nproc_per_node=1 --master_port=${port} \
              sample_brain_va.py \
              --base ${BASE_ARGS} \
              --seed 42 --jsonpath ${split} --output_dir ${OUT} \
              > ${log} 2>&1 &
        echo "clamp=${name} half=${half} gpu=${gpu} port=${port} pid=$!" >> logs/pathB_clamp_mini68.pids
    done
done

echo "=== launched 8 processes (4 configs × 2 GPUs) on gpu5 ==="
cat logs/pathB_clamp_mini68.pids
