#!/usr/bin/env bash
# cont500 iter 300 (val-loss-best) - 540 sample 4-way inference on gpu5 GPU 0-3
set -u
PROJ=/public/home/maoyaoxin/zhangt/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
MODEL=configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml
INFER=configs/sf_v1/infer_pathB_cont500_iter300.yaml
NAME=pathB_cont500_iter300
OUTDIR=results/alpha_540/${NAME}

cd $PROJ
rm -rf ${OUTDIR}
mkdir -p ${OUTDIR} logs
: > logs/cont500_iter300_4way.pids

export CUDA_HOME=/public/home/maoyaoxin/anaconda3/envs/cinebrain
export PYTHONPATH=${PROJ}:${PYTHONPATH:-}

for i in 0 1 2 3; do
    gpu=${i}
    split=${i}
    log=logs/cont500_iter300_gpu5_gpu${gpu}.log
    port=$((29860 + i))
    jsonpath=/public/home/maoyaoxin/zhangt/xxt/datasets/full540_4split${split}.json

    CUDA_VISIBLE_DEVICES=${gpu} \
      nohup ${PY} -m torch.distributed.run \
          --standalone --nproc_per_node=1 --master_port=${port} \
          sample_brain_va.py \
          --base ${MODEL} ${INFER} \
          --seed 42 --jsonpath ${jsonpath} --output_dir ${OUTDIR} \
          > ${log} 2>&1 &
    echo "${NAME} gpu5 gpu${gpu} split${split} pid=$!" >> logs/cont500_iter300_4way.pids
done
echo '=== iter 300 launched ==='
cat logs/cont500_iter300_4way.pids
