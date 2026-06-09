#!/bin/bash
# Fine-tune checkpoint_2 on tasks 1,7,18,21
# 用法: bash launch_finetune.sh [config_name]
# config_name 選項: ft_ckpt2_tasks1_7_18_21 | rft_task18_tidying_bedroom

WORK="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution"
NAS="/media/ML_2025/shawn/b1k"
PLUTO="/media/Pluto/Shawn/NTHU_Course_1142/b1k"
PYTHON="$NAS/venv/bin/python"
CONFIG="${1:-ft_ckpt2_tasks1_7_18_21}"

export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
# 只用 GPU2（GPU1 跑 OmniGibson eval 用）
export CUDA_VISIBLE_DEVICES=2
export JAX_PLATFORMS=cuda
# Make sure we use the correct venv
unset VIRTUAL_ENV
# Redirect HuggingFace cache to Pluto (/ disk is full)
export HF_HOME="/media/Pluto/Shawn/NTHU_Course_1142/b1k/hf_cache"
export HF_DATASETS_CACHE="/media/Pluto/Shawn/NTHU_Course_1142/b1k/hf_cache/datasets"
mkdir -p "$HF_DATASETS_CACHE"

LOG="$PLUTO/logs/finetune_${CONFIG}.log"
mkdir -p "$PLUTO/logs"

echo "[$(date '+%H:%M:%S')] 啟動 fine-tune: $CONFIG 在 GPU2"
echo "[$(date '+%H:%M:%S')] 使用 Python: $PYTHON"
echo "[$(date '+%H:%M:%S')] Log: $LOG"

cd "$WORK"
nohup "$PYTHON" -u scripts/train.py "$CONFIG" > "$LOG" 2>&1 &
FT_PID=$!
echo "PID: $FT_PID"
echo "$FT_PID" > "$PLUTO/logs/finetune_pid.txt"
echo "Fine-tune 已在背景啟動。用 tail -f $LOG 監控進度。"
