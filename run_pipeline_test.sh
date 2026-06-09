#!/bin/bash
# 2-trial pipeline test：驗證 OmniGibson + policy server 整合是否正常
set -e
WORK="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution"
NAS="/media/ML_2025/shawn/b1k"
PLUTO="/media/Pluto/Shawn/NTHU_Course_1142/b1k"
OG_DATA="/media/public_dataset2/behavior-1k/omnigibson_data"
CKPT="${1:-$NAS/checkpoints/checkpoint_1/checkpoint_1}"

export UV_PROJECT_ENVIRONMENT="$NAS/venv"
export OMNIGIBSON_DATA_PATH="$OG_DATA"
export OMNIGIBSON_APPDATA_PATH="$PLUTO/outputs/og_appdata"
source ~/.zshrc 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OMNIGIBSON_APPDATA_PATH" "$PLUTO/logs"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PLUTO/logs/pipeline_test.log"; }
log "=== Pipeline Test 開始（2 trials, task1）==="
log "Checkpoint: $CKPT"
log "Policy GPU: GPU2（${2:-2}）, Sim GPU: GPU0（0）"

POLICY_GPU="${2:-2}"
SIM_GPU=0

# 啟動 policy server on GPU${POLICY_GPU}
log "啟動 policy server on GPU${POLICY_GPU}..."
CUDA_VISIBLE_DEVICES=$POLICY_GPU XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
  "$NAS/venv/bin/python" "$WORK/scripts/serve_b1k.py" \
    --task-checkpoint-mapping "$WORK/task_checkpoint_mapping.json" \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir "$CKPT" \
    2>&1 | tee "$PLUTO/logs/policy_server_test.log" &
SERVER_PID=$!
log "Policy server PID: $SERVER_PID，等待啟動（30秒）..."
sleep 35

# 跑 2 trials
log "跑 2 trials on task1（GPU${SIM_GPU} 跑 OmniGibson）..."
OMNIGIBSON_DATA_PATH="$OG_DATA" \
OMNIGIBSON_APPDATA_PATH="$OMNIGIBSON_APPDATA_PATH" \
OMNIGIBSON_GPU_ID=$SIM_GPU \
  python "$WORK/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py" \
    log_path="$PLUTO/outputs/pipeline_test" \
    policy=websocket \
    task.name=picking_up_trash \
    model.host=localhost \
    eval_instance_ids="[0,1]" \
    2>&1 | tee "$PLUTO/logs/pipeline_test_eval.log"

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
SUCCESS=$(grep -oP "success_rate[:\s]+\K[\d.]+" "$PLUTO/logs/pipeline_test_eval.log" | tail -1 || echo "N/A")
log "=== Test 完成！success_rate=$SUCCESS ==="
echo "PIPELINE_TEST: success=$SUCCESS" >> "$PLUTO/logs/all_results.txt"
