#!/usr/bin/env bash
# Path B P1 iter 500 - 540 sample 8-way inference on gpu5.
# Goal: verify diagnostic claim that iter 500 FVD << iter 2000 FVD 517.
# val_loss 0.396 (iter 500) vs 0.486 (iter 2000) predicts ~425 FVD.
# Wall clock: 540 / 8 * 4.5 min ~= 5.1h (~8.6 min/sample with FORCE_DETERMINISM off).

set -u
PROJ=/public/home/maoyaoxin/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
MODEL=configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml
INFER=configs/sf_v1/infer_pathB_p1_iter500.yaml
NAME=pathB_p1_iter500
OUTDIR=results/alpha_540/${NAME}

cd $PROJ
rm -rf ${OUTDIR}
mkdir -p ${OUTDIR} logs
: > logs/pathB_iter500_8way.pids

# gpu5-specific env: no /usr/local/cuda-12.4; use conda env cuda
export CUDA_HOME=/public/home/maoyaoxin/anaconda3/envs/cinebrain
export PYTHONPATH=${PROJ}:${PYTHONPATH:-}

for gpu in 0 1 2 3 4 5 6 7; do
    split=${gpu}
    log=logs/pathB_iter500_gpu5_gpu${gpu}.log
    port=$((29840 + gpu))
    jsonpath=/public/home/maoyaoxin/xxt/datasets/full540_8split${split}.json

    CUDA_VISIBLE_DEVICES=${gpu} \
      nohup ${PY} -m torch.distributed.run \
          --standalone --nproc_per_node=1 --master_port=${port} \
          sample_brain_va.py \
          --base ${MODEL} ${INFER} \
          --seed 42 --jsonpath ${jsonpath} --output_dir ${OUTDIR} \
          > ${log} 2>&1 &
    echo "${NAME} gpu5 gpu${gpu} split${split} pid=$!" >> logs/pathB_iter500_8way.pids
done

echo "=== launched 8 processes on gpu5 ==="
cat logs/pathB_iter500_8way.pids
echo ""
echo "Monitor with:"
echo "  ssh gpu5 \"ls ${OUTDIR}/*.mp4 2>/dev/null | wc -l\"  # expect 540 at completion"
echo "  tail -f logs/pathB_iter500_gpu5_gpu0.log"
