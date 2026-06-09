#!/bin/bash
# Phase 3: Final Evaluation，比較 baseline vs fine-tuned
set -e
WORK_DIR="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution"
NAS_DIR="/media/ML_2025/shawn/b1k"
OG_DATA="/media/ML_2025/shawn/b1k/og_assets"
export UV_PROJECT_ENVIRONMENT="$NAS_DIR/venv"
source ~/.zshrc 2>/dev/null || true

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 確認 OmniGibson 資料已掛載 ──────────────────────────────────────
if [ ! -d "$OG_DATA" ]; then
    log "ERROR: OmniGibson assets 尚未下載，請等待 setup_and_download.sh 的 Step 9 完成"
    exit 1
fi
export OMNIGIBSON_DATA_PATH="$OG_DATA"
export OMNIGIBSON_APPDATA_PATH="$NAS_DIR/outputs/og_appdata"
mkdir -p "$OMNIGIBSON_APPDATA_PATH"
log "OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH"

RESULT_FILE="$NAS_DIR/logs/final_eval_results.txt"
echo "=== Final Evaluation $(date) ===" > "$RESULT_FILE"

# Fine-tuned checkpoint 路徑（取最新的 step 目錄）
find_latest_ckpt() {
    local base_dir="$1"
    ls -d "${base_dir}"/step_* 2>/dev/null | sort -V | tail -1
}

FT_CKPT2=$(find_latest_ckpt "$NAS_DIR/outputs/checkpoints/pi_behavior_b1k_fast/finetune_ckpt2_tasks1_7_18_21")
FT_CKPT3=$(find_latest_ckpt "$NAS_DIR/outputs/checkpoints/pi_behavior_b1k_fast/finetune_ckpt3_task27")
FT_CKPT1=$(find_latest_ckpt "$NAS_DIR/outputs/checkpoints/pi_behavior_b1k_fast/finetune_ckpt1_task29")

log "Fine-tuned checkpoints:"
log "  Ckpt2 (tasks 1,7,18,21): $FT_CKPT2"
log "  Ckpt3 (task 27):          $FT_CKPT3"
log "  Ckpt1 (task 29):          $FT_CKPT1"

declare -A TASK_FTCKPT=(
    ["picking_up_trash"]="$FT_CKPT2"
    ["picking_up_toys"]="$FT_CKPT2"
    ["tidying_bedroom"]="$FT_CKPT2"
    ["collecting_childrens_toys"]="$FT_CKPT2"
    ["sorting_household_items"]="$FT_CKPT3"
    ["clean_up_your_desk"]="$FT_CKPT1"
)

for TASK in "${!TASK_FTCKPT[@]}"; do
    CKPT_PATH="${TASK_FTCKPT[$TASK]}"
    if [ -z "$CKPT_PATH" ]; then
        log "WARNING: $TASK 的 fine-tuned checkpoint 不存在，跳過"
        continue
    fi
    log "評估 fine-tuned: $TASK (ckpt: $CKPT_PATH)"

    CUDA_VISIBLE_DEVICES=1 uv run "$WORK_DIR/scripts/serve_b1k.py" \
        --task-checkpoint-mapping "$WORK_DIR/task_checkpoint_mapping.json" \
        policy:checkpoint \
        --policy.config pi_behavior_b1k_fast \
        --policy.dir "$CKPT_PATH" &
    SERVER_PID=$!
    sleep 20

    EVAL_LOG="$NAS_DIR/logs/final_${TASK}.log"
    OMNIGIBSON_DATA_PATH="$OG_DATA" \
    OMNIGIBSON_APPDATA_PATH="$OMNIGIBSON_APPDATA_PATH" \
    python "$WORK_DIR/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py" \
        log_path="$NAS_DIR/outputs/final_eval" \
        policy=websocket \
        task.name="$TASK" \
        model.host=localhost \
        eval_instance_ids="[0,1,2,3]" \
        2>&1 | tee "$EVAL_LOG"

    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 5

    SUCCESS_RATE=$(grep -oP "success_rate[:\s]+\K[\d.]+" "$EVAL_LOG" | tail -1 || echo "N/A")
    BASELINE=$(grep "BASELINE $TASK" "$NAS_DIR/logs/baseline_eval_results.txt" 2>/dev/null | awk '{print $NF}' || echo "N/A")
    log "Task $TASK: baseline=$BASELINE -> fine-tuned=$SUCCESS_RATE"
    echo "FINETUNED $TASK : baseline=$BASELINE -> ft=$SUCCESS_RATE" >> "$RESULT_FILE"
done

echo "" >> "$RESULT_FILE"
echo "=== 比較摘要 ===" >> "$RESULT_FILE"
grep "FINETUNED" "$RESULT_FILE" | sort
log "=== 最終評估完成，結果存於 $RESULT_FILE ==="
cat "$RESULT_FILE"
