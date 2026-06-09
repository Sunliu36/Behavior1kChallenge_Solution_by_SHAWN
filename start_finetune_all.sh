#!/bin/bash
# 三卡並行 fine-tune：等資料 + checkpoints 都就緒後立刻啟動
# GPU0: ft_ckpt1_task29  (task 29，小任務，batch=8 適配 GR00T 佔用)
# GPU1: ft_ckpt2_tasks1_7_18_21  (tasks 1,7,18,21，主力)
# GPU2: ft_ckpt3_task27  (task 27)
set -e
PROJ="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution"
NAS="/media/ML_2025/shawn/b1k"
export UV_PROJECT_ENVIRONMENT="$NAS/venv"
export PATH="$HOME/.local/bin:$PATH"
source ~/.zshrc 2>/dev/null || true

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── 等待資料就緒 ──────────────────────────────────────────────────────
log "等待 checkpoints 和訓練資料就緒..."
wait_ready() {
    while true; do
        local ckpt1="$NAS/checkpoints/checkpoint_1/checkpoint_1/params/_METADATA"
        local ckpt2="$NAS/checkpoints/checkpoint_2/checkpoint_2/params/_METADATA"
        local ckpt3="$NAS/checkpoints/checkpoint_3/checkpoint_3/params/_METADATA"
        local data="$NAS/data/behavior_224_rgb/data/task-0001"
        local missing=0
        for f in "$ckpt1" "$ckpt2" "$ckpt3" "$data"; do
            [ ! -e "$f" ] && log "等待: $f" && missing=1
        done
        [ $missing -eq 0 ] && break
        sleep 15
    done
    log "所有資料就緒，開始 fine-tune！"
}
wait_ready

# ── 確認 assets（norm stats）已就緒 ──────────────────────────────────
ASSETS_DIR="$NAS/outputs/assets/ft_ckpt2_tasks1_7_18_21"
mkdir -p "$ASSETS_DIR"
if [ ! -f "$ASSETS_DIR/norm_stats.json" ] && [ ! -f "$ASSETS_DIR/norm_stats.pkl" ]; then
    log "複製 norm stats 從 checkpoint_2..."
    SRC="$NAS/checkpoints/checkpoint_2/checkpoint_2/assets"
    [ -d "$SRC" ] && cp -rn "$SRC/"* "$NAS/outputs/assets/" 2>/dev/null || true
fi
# 同樣處理 ckpt3 / ckpt1
for ckpt_name in ft_ckpt3_task27 ft_ckpt1_task29; do
    ADIR="$NAS/outputs/assets/$ckpt_name"
    mkdir -p "$ADIR"
done

log "=== 啟動三卡並行 fine-tune ==="

# ── GPU1：ft_ckpt2（tasks 1,7,18,21）主力訓練 ─────────────────────
log "GPU1: ft_ckpt2_tasks1_7_18_21 (10000 steps, batch=16)"
CUDA_VISIBLE_DEVICES=1 uv run scripts/train.py ft_ckpt2_tasks1_7_18_21 \
    --batch_size=16 \
    --num_train_steps=10000 \
    --save_interval=500 \
    --keep_period=2000 \
    --log_interval=25 \
    --overwrite \
    2>&1 | tee "$NAS/logs/ft_ckpt2.log" &
GPU1_PID=$!
log "GPU1 PID: $GPU1_PID"

# 稍等 3 秒讓 GPU1 先佔住 JAX 初始化
sleep 3

# ── GPU2：ft_ckpt3（task 27）─────────────────────────────────────
log "GPU2: ft_ckpt3_task27 (5000 steps, batch=16)"
CUDA_VISIBLE_DEVICES=2 uv run scripts/train.py ft_ckpt3_task27 \
    --batch_size=16 \
    --num_train_steps=5000 \
    --save_interval=500 \
    --keep_period=2000 \
    --log_interval=25 \
    --overwrite \
    2>&1 | tee "$NAS/logs/ft_ckpt3.log" &
GPU2_PID=$!
log "GPU2 PID: $GPU2_PID"

sleep 3

# ── GPU0：ft_ckpt1（task 29）batch 縮小適配 GR00T 佔用 ─────────────
GPU0_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | head -1)
log "GPU0 可用顯存: ${GPU0_FREE}MiB"
if [ "$GPU0_FREE" -gt 15000 ]; then
    log "GPU0: ft_ckpt1_task29 (5000 steps, batch=8)"
    CUDA_VISIBLE_DEVICES=0 uv run scripts/train.py ft_ckpt1_task29 \
        --batch_size=8 \
        --num_train_steps=5000 \
        --save_interval=500 \
        --keep_period=2000 \
        --log_interval=25 \
        --overwrite \
        2>&1 | tee "$NAS/logs/ft_ckpt1.log" &
    GPU0_PID=$!
    log "GPU0 PID: $GPU0_PID"
else
    log "GPU0 顯存不足（${GPU0_FREE}MiB），跳過；ft_ckpt1 將在 GPU2 完成後接著跑"
    GPU0_PID=""
fi

# ── 等待所有訓練完成 ──────────────────────────────────────────────
log "等待 GPU1 ft_ckpt2 完成..."
wait $GPU1_PID && log "GPU1 ft_ckpt2 完成 ✓" || log "GPU1 ft_ckpt2 異常結束，請檢查 ft_ckpt2.log"

log "等待 GPU2 ft_ckpt3 完成..."
wait $GPU2_PID && log "GPU2 ft_ckpt3 完成 ✓" || log "GPU2 ft_ckpt3 異常結束"

# 如果 GPU0 沒跑 ckpt1，在 GPU2 完成後補跑
if [ -z "$GPU0_PID" ]; then
    log "GPU2: 接著跑 ft_ckpt1_task29 (5000 steps)"
    CUDA_VISIBLE_DEVICES=2 uv run scripts/train.py ft_ckpt1_task29 \
        --batch_size=16 \
        --num_train_steps=5000 \
        --save_interval=500 \
        --keep_period=2000 \
        --log_interval=25 \
        --overwrite \
        2>&1 | tee "$NAS/logs/ft_ckpt1.log"
    log "ft_ckpt1 完成 ✓"
else
    wait $GPU0_PID && log "GPU0 ft_ckpt1 完成 ✓" || log "GPU0 ft_ckpt1 異常結束"
fi

log "=== 所有 fine-tune 完成 ==="
touch "$NAS/logs/finetune_done.sentinel"
log "Sentinel: $NAS/logs/finetune_done.sentinel"
