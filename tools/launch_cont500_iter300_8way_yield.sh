#!/usr/bin/env bash
# cont500 iter 300 - 540 sample 8-way inference on gpu5 with COURTESY YIELD watchdog.
# - skip-if-exists: leverages sample_brain_va.py:167 to resume from existing mp4.
# - watchdog: every 60s, checks nvidia-smi compute-apps for alien PIDs.
#   On detection: SIGTERM all our workers, exit. Skip-if-exists handles relaunch.

set -u
PROJ=/public/home/maoyaoxin/zhangt/xxt/SF-v3
PY=/public/home/maoyaoxin/anaconda3/envs/cinebrain/bin/python
MODEL=configs/sf_v1/cinebrain_sf_v3_pathB_model.yaml
INFER=configs/sf_v1/infer_pathB_cont500_iter300.yaml
NAME=pathB_cont500_iter300
OUTDIR=results/alpha_540/${NAME}

cd $PROJ
mkdir -p ${OUTDIR} logs
: > logs/cont500_iter300_8way_yield.pids

export CUDA_HOME=/public/home/maoyaoxin/anaconda3/envs/cinebrain
export PYTHONPATH=${PROJ}:${PYTHONPATH:-}

START_TS=$(date +%s)
echo "=== launch $(date) ==="
echo "already done: $(ls ${OUTDIR}/*.mp4 2>/dev/null | wc -l)/540"

WORKER_PIDS=()
for gpu in 0 1 2 3 4 5 6 7; do
    split=${gpu}
    log=logs/cont500_iter300_yield_gpu${gpu}.log
    port=$((29880 + gpu))
    jsonpath=/public/home/maoyaoxin/zhangt/xxt/datasets/full540_8split${split}.json
    CUDA_VISIBLE_DEVICES=${gpu} \
      nohup ${PY} -m torch.distributed.run \
          --standalone --nproc_per_node=1 --master_port=${port} \
          sample_brain_va.py \
          --base ${MODEL} ${INFER} \
          --seed 42 --jsonpath ${jsonpath} --output_dir ${OUTDIR} \
          > ${log} 2>&1 &
    WORKER_PIDS+=($!)
    echo "${NAME} gpu${gpu} split${split} pid=$!" >> logs/cont500_iter300_8way_yield.pids
done
echo "=== launched 8 workers, PIDs: ${WORKER_PIDS[*]} ==="

WATCHDOG_LOG=logs/cont500_iter300_watchdog.log
echo "[watchdog] started $(date)" > ${WATCHDOG_LOG}

is_alien() {
    local pid=$1
    [ -z "$pid" ] && return 1
    local cmd
    cmd=$(ps -o cmd= -p $pid 2>/dev/null)
    [ -z "$cmd" ] && return 1
    case "$cmd" in
        *sample_brain_va*) return 1 ;;
        *torch.distributed.run*) return 1 ;;
        *) return 0 ;;
    esac
}

check_contention() {
    local pids
    for gpu in 0 1 2 3 4 5 6 7; do
        pids=$(nvidia-smi -i $gpu --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
        for pid in $pids; do
            if is_alien "$pid"; then
                local cmd
                cmd=$(ps -o cmd= -p $pid 2>/dev/null | head -c 80)
                echo "[watchdog] $(date) ALIEN gpu${gpu} pid=$pid cmd=$cmd" >> ${WATCHDOG_LOG}
                return 0
            fi
        done
    done
    return 1
}

SLEEP=60
while true; do
    alive=0
    for pid in "${WORKER_PIDS[@]}"; do
        kill -0 $pid 2>/dev/null && alive=$((alive+1))
    done
    if [ $alive -eq 0 ]; then
        echo "[watchdog] $(date) all workers exited (done or crashed)" >> ${WATCHDOG_LOG}
        break
    fi

    n_mp4=$(ls ${OUTDIR}/*.mp4 2>/dev/null | wc -l)
    if [ $n_mp4 -ge 540 ]; then
        echo "[watchdog] $(date) reached 540 mp4 — complete" >> ${WATCHDOG_LOG}
        break
    fi

    if check_contention; then
        echo "[watchdog] $(date) YIELDING — SIGTERM to workers" >> ${WATCHDOG_LOG}
        for pid in "${WORKER_PIDS[@]}"; do
            kill -TERM $pid 2>/dev/null
            pkill -TERM -P $pid 2>/dev/null
        done
        sleep 30
        for pid in "${WORKER_PIDS[@]}"; do
            kill -KILL $pid 2>/dev/null
            pkill -KILL -P $pid 2>/dev/null
        done
        break
    fi

    sleep $SLEEP
done

END_TS=$(date +%s)
DUR=$((END_TS - START_TS))
echo "[watchdog] $(date) end. duration=${DUR}s mp4=$(ls ${OUTDIR}/*.mp4 2>/dev/null | wc -l)/540" >> ${WATCHDOG_LOG}
