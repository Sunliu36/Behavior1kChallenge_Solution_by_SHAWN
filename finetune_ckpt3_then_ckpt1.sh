#!/bin/bash
# Phase 2: Fine-tune Checkpoint 3（task 27）→ 接著 Checkpoint 1（task 29）on GPU2
set -e
WORK_DIR="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution"
NAS_DIR="/media/ML_2025/shawn/b1k"
export UV_PROJECT_ENVIRONMENT="$NAS_DIR/venv"
source ~/.zshrc 2>/dev/null || true

log() { echo "[$(date '+%H:%M:%S')] $*"; }
OUTPUT_DIR="$NAS_DIR/outputs/checkpoints"
ASSETS_DIR="$NAS_DIR/outputs/assets"
DATA_DIR="$NAS_DIR/data/behavior_224_rgb"

cd "$WORK_DIR"

# ── Checkpoint 3: task 27（sorting_household_items）──────────────────
log "=== Fine-tune Checkpoint 3 開始（GPU2，task 27）==="
CKPT3="$NAS_DIR/checkpoints/checkpoint_3"

CUDA_VISIBLE_DEVICES=2 uv run scripts/train.py pi_behavior_b1k_fast \
    --batch_size=16 \
    --num_train_steps=5000 \
    --save_interval=1000 \
    --keep_period=2000 \
    --log_interval=50 \
    --exp_name="finetune_ckpt3_task27" \
    --data.base_config.behavior_dataset_root="$DATA_DIR" \
    --assets_base_dir="$ASSETS_DIR" \
    --checkpoint_base_dir="$OUTPUT_DIR" \
    --weight_loader.checkpoint_path="$CKPT3" \
    2>&1 | tee "$NAS_DIR/logs/finetune_ckpt3.log"

log "=== Checkpoint 3 Fine-tune 完成 ==="

# ── Checkpoint 1: task 29（clean_up_your_desk）───────────────────────
log "=== Fine-tune Checkpoint 1 開始（GPU2，task 29）==="
CKPT1="$NAS_DIR/checkpoints/checkpoint_1"

CUDA_VISIBLE_DEVICES=2 uv run scripts/train.py pi_behavior_b1k_fast \
    --batch_size=16 \
    --num_train_steps=5000 \
    --save_interval=1000 \
    --keep_period=2000 \
    --log_interval=50 \
    --exp_name="finetune_ckpt1_task29" \
    --data.base_config.behavior_dataset_root="$DATA_DIR" \
    --assets_base_dir="$ASSETS_DIR" \
    --checkpoint_base_dir="$OUTPUT_DIR" \
    --weight_loader.checkpoint_path="$CKPT1" \
    2>&1 | tee "$NAS_DIR/logs/finetune_ckpt1.log"

log "=== Checkpoint 1 Fine-tune 完成 ==="
log "=== Phase 2 全部完成，請執行 run_final_eval.sh ==="
