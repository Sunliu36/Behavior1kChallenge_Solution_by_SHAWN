#!/bin/bash
# Phase 2: Fine-tune Checkpoint 2（tasks 1, 7, 18, 21）on GPU1
# 接在 setup_and_download.sh 完成後執行
set -e
WORK_DIR="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution"
NAS_DIR="/media/ML_2025/shawn/b1k"
export UV_PROJECT_ENVIRONMENT="$NAS_DIR/venv"
source ~/.zshrc 2>/dev/null || true

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Fine-tune Checkpoint 2 開始（GPU1）==="
log "Tasks: picking_up_trash, picking_up_toys, tidying_bedroom, collecting_childrens_toys"

CKPT2="$NAS_DIR/checkpoints/checkpoint_2"
OUTPUT_DIR="$NAS_DIR/outputs/checkpoints"
ASSETS_DIR="$NAS_DIR/outputs/assets"
DATA_DIR="$NAS_DIR/data/behavior_224_rgb"

# 確認 checkpoint 存在
if [ ! -d "$CKPT2" ]; then
    log "ERROR: Checkpoint 2 不存在於 $CKPT2"
    exit 1
fi

cd "$WORK_DIR"

# 修改 config 讓 data path 指向我們的 NAS 資料
# 使用 override 方式傳入路徑參數
CUDA_VISIBLE_DEVICES=1 uv run scripts/train.py pi_behavior_b1k_fast \
    --batch_size=16 \
    --num_train_steps=10000 \
    --save_interval=1000 \
    --keep_period=2000 \
    --log_interval=50 \
    --exp_name="finetune_ckpt2_tasks1_7_18_21" \
    --data.base_config.behavior_dataset_root="$DATA_DIR" \
    --assets_base_dir="$ASSETS_DIR" \
    --checkpoint_base_dir="$OUTPUT_DIR" \
    --weight_loader.checkpoint_path="$CKPT2" \
    2>&1 | tee "$NAS_DIR/logs/finetune_ckpt2.log"

log "=== Checkpoint 2 Fine-tune 完成 ==="
