#!/bin/bash
# GPU 1 上跑 task 27 fine-tune
# 用法: bash launch_finetune_gpu1.sh [config_name]

WORK="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution"
NAS="/media/ML_2025/shawn/b1k"
PLUTO="/media/Pluto/Shawn/NTHU_Course_1142/b1k"
PYTHON="$NAS/venv/bin/python"
CONFIG="${1:-ft_ckpt3_task27}"

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
# 只用 GPU1
export CUDA_VISIBLE_DEVICES=1
export JAX_PLATFORMS=cuda
unset VIRTUAL_ENV
export HF_HOME="/media/Pluto/Shawn/NTHU_Course_1142/b1k/hf_cache"
export HF_DATASETS_CACHE="/media/Pluto/Shawn/NTHU_Course_1142/b1k/hf_cache/datasets"
mkdir -p "$HF_DATASETS_CACHE"

LOG="$PLUTO/logs/finetune_gpu1_${CONFIG}.log"
mkdir -p "$PLUTO/logs"

# Clean up old checkpoint dir
rm -rf "$PLUTO/outputs/checkpoints/${CONFIG}"

echo "[$(date '+%H:%M:%S')] 啟動 GPU1 fine-tune: $CONFIG"
echo "[$(date '+%H:%M:%S')] Log: $LOG"

cd "$WORK"
nohup "$PYTHON" -u scripts/train.py "$CONFIG" > "$LOG" 2>&1 &
GPU1_PID=$!
echo "PID: $GPU1_PID"
echo "$GPU1_PID" > "$PLUTO/logs/finetune_gpu1_pid.txt"
echo "GPU1 Fine-tune 已在背景啟動。"
