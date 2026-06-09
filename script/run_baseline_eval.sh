#!/bin/bash
# Baseline Evaluation：6 個 target tasks，各跑 4 個 instance
set -e
WORK_DIR="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution"
NAS_DIR="/media/ML_2025/shawn/b1k"
OG_DATA="/media/ML_2025/shawn/b1k/og_assets"
export UV_PROJECT_ENVIRONMENT="$NAS_DIR/venv"
source ~/.zshrc 2>/dev/null || true

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 確認 OmniGibson 資料已掛載 ──────────────────────────────────────
if [ ! -d "$OG_DATA" ]; then
    log "ERROR: OmniGibson 資料路徑不存在: $OG_DATA"
    log "請先執行: sudo mount -t nfs -o port=2050 140.114.27.95:/volume1/public_dataset2 /media/public_dataset2"
    exit 1
fi
# OmniGibson 必要的環境變數
export OMNIGIBSON_DATA_PATH="$OG_DATA"
export OMNIGIBSON_APPDATA_PATH="$NAS_DIR/outputs/og_appdata"  # 把 Omniverse cache 放 NAS 避免 NVMe 滿
mkdir -p "$OMNIGIBSON_APPDATA_PATH"
log "OMNIGIBSON_DATA_PATH=$OMNIGIBSON_DATA_PATH"

# 確認必要子目錄存在
if [ ! -d "$OG_DATA/2025-challenge-task-instances" ]; then
    log "ERROR: 找不到 2025-challenge-task-instances，請確認 omnigibson_data 結構"
    log "目錄內容: $(ls $OG_DATA)"
    exit 1
fi

# Target tasks（task_name 對應模擬器的名稱）
declare -A TASK_CKPT=(
    ["picking_up_trash"]="checkpoint_2"
    ["picking_up_toys"]="checkpoint_2"
    ["tidying_bedroom"]="checkpoint_2"
    ["collecting_childrens_toys"]="checkpoint_2"
    ["sorting_household_items"]="checkpoint_3"
    ["clean_up_your_desk"]="checkpoint_1"
)

RESULT_FILE="$NAS_DIR/logs/baseline_eval_results.txt"
echo "=== Baseline Evaluation $(date) ===" > "$RESULT_FILE"

for TASK in "${!TASK_CKPT[@]}"; do
    CKPT="${TASK_CKPT[$TASK]}"
    CKPT_PATH="$NAS_DIR/checkpoints/$CKPT"
    log "評估 task: $TASK (使用 $CKPT)"

    # 1. 啟動 policy server（GPU0 以外，指定 GPU1）
    CUDA_VISIBLE_DEVICES=1 uv run "$WORK_DIR/scripts/serve_b1k.py" \
        --task-checkpoint-mapping "$WORK_DIR/task_checkpoint_mapping.json" \
        policy:checkpoint \
        --policy.config pi_behavior_b1k_fast \
        --policy.dir "$CKPT_PATH" &
    SERVER_PID=$!
    sleep 20  # 等 server 啟動（第一次載入模型需要較久）

    # 2. 執行 eval（使用 GPU0 跑 OmniGibson 模擬）
    EVAL_LOG="$NAS_DIR/logs/baseline_${TASK}.log"
    OMNIGIBSON_DATA_PATH="$OG_DATA" \
    OMNIGIBSON_APPDATA_PATH="$OMNIGIBSON_APPDATA_PATH" \
    python "$WORK_DIR/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py" \
        log_path="$NAS_DIR/outputs/baseline_eval" \
        policy=websocket \
        task.name="$TASK" \
        model.host=localhost \
        eval_instance_ids="[0,1,2,3]" \
        2>&1 | tee "$EVAL_LOG"

    # 3. 關閉 server
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 5

    # 4. 擷取結果
    SUCCESS_RATE=$(grep -oP "success_rate[:\s]+\K[\d.]+" "$EVAL_LOG" | tail -1 || echo "N/A")
    log "Task $TASK: success_rate = $SUCCESS_RATE"
    echo "BASELINE $TASK : $SUCCESS_RATE" >> "$RESULT_FILE"
done

log "=== Baseline Evaluation 完成，結果存於 $RESULT_FILE ==="
cat "$RESULT_FILE"
