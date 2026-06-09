#!/bin/bash
# Fine-tune 完成後自動更新 checkpoint mapping、重啟 policy server、執行 final eval
# 用法: nohup bash auto_post_finetune.sh > /tmp/auto_post_ft.log 2>&1 &

PLUTO="/media/Pluto/Shawn/NTHU_Course_1142/b1k"
WORK="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd"
FT_LOG="$PLUTO/logs/finetune_ft_ckpt2_tasks1_7_18_21.log"
FT_PID_FILE="$PLUTO/logs/finetune_pid.txt"

echo "[$(date '+%H:%M:%S')] 等待 fine-tuning 完成..."

# Read PID
wait_for_ft() {
    local pid=""
    # Wait until PID file exists
    while [ -z "$pid" ]; do
        if [ -f "$FT_PID_FILE" ]; then
            pid=$(cat "$FT_PID_FILE")
        fi
        [ -z "$pid" ] && sleep 30
    done
    echo "$pid"
}

FT_PID=$(wait_for_ft)
echo "[$(date '+%H:%M:%S')] Fine-tune PID: $FT_PID"

# Monitor progress
LAST_STEP=0
STALL_COUNT=0
while ps -p $FT_PID > /dev/null 2>&1; do
    sleep 300  # check every 5 minutes
    # Get latest step from log
    STEP=$(grep -oP "train_step=\K\d+" "$FT_LOG" 2>/dev/null | tail -1 || echo "?")
    CKPT_LATEST=$(ls "$PLUTO/outputs/checkpoints/ft_ckpt2_tasks1_7_18_21/ft_ckpt2/" 2>/dev/null | grep "^[0-9]" | sort -n | tail -1 || echo "none")
    echo "[$(date '+%H:%M:%S')] step=$STEP latest_ckpt=$CKPT_LATEST"

    # Stall detection: if step hasn't changed for 30 min, log warning
    if [ "$STEP" = "$LAST_STEP" ] && [ "$STEP" != "?" ]; then
        STALL_COUNT=$((STALL_COUNT + 1))
        [ $STALL_COUNT -ge 6 ] && echo "[$(date '+%H:%M:%S')] WARNING: training stalled at step=$STEP for 30 min"
    else
        STALL_COUNT=0
        LAST_STEP="$STEP"
    fi
done

echo "[$(date '+%H:%M:%S')] Fine-tuning 完成！"

# Check what checkpoint was saved
CKPT_DIR="$PLUTO/outputs/checkpoints/ft_ckpt2_tasks1_7_18_21/ft_ckpt2"
LATEST_STEP=$(ls "$CKPT_DIR/" 2>/dev/null | grep "^[0-9]" | sort -n | tail -1)
echo "[$(date '+%H:%M:%S')] 最新 checkpoint: step=$LATEST_STEP"

if [ -z "$LATEST_STEP" ]; then
    echo "ERROR: No checkpoint found at $CKPT_DIR"
    exit 1
fi

# Update checkpoint mapping and restart policy server
echo "[$(date '+%H:%M:%S')] 更新 checkpoint mapping 並重啟 policy server..."
bash "$WORK/update_checkpoint_mapping.sh" ft_ckpt2
if [ $? -ne 0 ]; then
    echo "ERROR: update_checkpoint_mapping.sh failed!"
    exit 1
fi

echo "[$(date '+%H:%M:%S')] 等待 policy server 穩定 (60s)..."
sleep 60

# Verify policy server is running
if ! ps aux | grep "serve_b1k.py" | grep -v grep > /dev/null; then
    echo "ERROR: Policy server not running after restart!"
    exit 1
fi

# Run final eval for all 6 tasks
echo "[$(date '+%H:%M:%S')] 開始 final eval (6 tasks × 2 instances)..."
bash "$WORK/run_all_evals.sh" ft_eval 2 2>&1 | tee "$PLUTO/logs/final_eval_all_tasks.log"

echo "[$(date '+%H:%M:%S')] === 全部完成 ==="
echo "Fine-tune + Eval pipeline 已完成。"
echo "結果在: $PLUTO/outputs/ft_eval_*"
