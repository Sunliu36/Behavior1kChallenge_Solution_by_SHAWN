#!/bin/bash
# Task 1 Evaluation：10 trials
# 用法: bash run_eval_task1.sh [checkpoint_dir] [tag]
# 例如: bash run_eval_task1.sh /media/Pluto/Shawn/NTHU_Course_1142/b1k/outputs/checkpoints/rft_task1_picking_up_trash/rft_task1 "sft"

set -e
WORK="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution"
NAS="/media/ML_2025/shawn/b1k"
PLUTO="/media/Pluto/Shawn/NTHU_Course_1142/b1k"
OG_DATA="/media/public_dataset2/behavior-1k/omnigibson_data"

CKPT_DIR="${1:-$NAS/checkpoints/checkpoint_1/checkpoint_1}"
TAG="${2:-baseline}"

export UV_PROJECT_ENVIRONMENT="$NAS/venv"
export OMNIGIBSON_DATA_PATH="$OG_DATA"
export OMNIGIBSON_APPDATA_PATH="$PLUTO/outputs/og_appdata"
export OMNIGIBSON_GPU_ID=0    # eval 用 GPU0
source ~/.zshrc 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OMNIGIBSON_APPDATA_PATH" "$PLUTO/logs"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "=== Task 1 Evaluation: tag=$TAG ==="
log "Checkpoint: $CKPT_DIR"
log "OMNIGIBSON_DATA_PATH: $OG_DATA"

RESULT_DIR="$PLUTO/outputs/eval_${TAG}_task1"
LOG_FILE="$PLUTO/logs/eval_${TAG}_task1.log"
mkdir -p "$RESULT_DIR"

# 1. 啟動 policy server（GPU1，inference 用）
log "啟動 policy server on GPU1..."
CUDA_VISIBLE_DEVICES=1 UV_PROJECT_ENVIRONMENT="$NAS/venv" \
XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
  uv run "$WORK/scripts/serve_b1k.py" \
    --task-checkpoint-mapping "$WORK/task_checkpoint_mapping.json" \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir "$CKPT_DIR" &
SERVER_PID=$!
log "Policy server PID: $SERVER_PID"
sleep 30  # 等 server 載入模型

# 2. 執行 eval（GPU0 跑 OmniGibson）
log "執行 10 trials on task: picking_up_trash..."
OMNIGIBSON_DATA_PATH="$OG_DATA" \
OMNIGIBSON_APPDATA_PATH="$OMNIGIBSON_APPDATA_PATH" \
OMNIGIBSON_GPU_ID=0 \
  python "$WORK/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py" \
    log_path="$RESULT_DIR" \
    policy=websocket \
    task.name=picking_up_trash \
    model.host=localhost \
    eval_instance_ids="[0,1,2,3,4,5,6,7,8,9]" \
    2>&1 | tee "$LOG_FILE"

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

# 3. 擷取結果
SUCCESS=$(grep -oP "success_rate[:\s]+\K[\d.]+" "$LOG_FILE" | tail -1 || echo "N/A")
QSCORE=$(grep -oP "q_score[:\s]+\K[\d.]+" "$LOG_FILE" | tail -1 || echo "N/A")
log "=== 結果 [${TAG}] task1: success_rate=$SUCCESS  q_score=$QSCORE ==="
echo "${TAG} task1: success=${SUCCESS} q_score=${QSCORE}" >> "$PLUTO/logs/all_results.txt"
