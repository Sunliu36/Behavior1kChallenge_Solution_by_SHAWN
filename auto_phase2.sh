#!/bin/bash
# 等 setup_and_download.sh 完成後，自動啟動 Phase 2 fine-tuning
# 由 tmux finetune_gpu1 和 finetune_gpu2 window 分別呼叫
PROJ="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd"
NAS="/media/ML_2025/shawn/b1k"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# 等待 setup 完成（檢查 sentinel 檔案）
log "等待 setup_and_download.sh 完成..."
while [ ! -f "$NAS/logs/setup_done.sentinel" ]; do
    sleep 30
    log "Setup 尚未完成，繼續等待..."
done
log "Setup 完成，開始 $1 ..."

case "$1" in
    gpu1)
        bash "$PROJ/finetune_ckpt2.sh"
        ;;
    gpu2)
        bash "$PROJ/finetune_ckpt3_then_ckpt1.sh"
        ;;
    baseline)
        bash "$PROJ/run_baseline_eval.sh"
        ;;
    *)
        log "用法: $0 [gpu1|gpu2|baseline]"
        exit 1
        ;;
esac
