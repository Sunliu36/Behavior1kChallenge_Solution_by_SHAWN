#!/bin/bash
# RFT Rollout：部署 policy server + 跑 perturbed rollouts + 篩選成功 trajectory
# 用法: bash run_rft_rollout.sh [checkpoint_dir] [num_rollouts]

set -e
WORK="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution"
NAS="/media/ML_2025/shawn/b1k"
PLUTO="/media/Pluto/Shawn/NTHU_Course_1142/b1k"
OG_DATA="/media/public_dataset2/behavior-1k/omnigibson_data"

CKPT_DIR="${1:-$PLUTO/outputs/checkpoints/rft_task1_picking_up_trash/rft_task1}"
NUM_ROLLOUTS="${2:-80}"

export UV_PROJECT_ENVIRONMENT="$NAS/venv"
export OMNIGIBSON_DATA_PATH="$OG_DATA"
export OMNIGIBSON_APPDATA_PATH="$PLUTO/outputs/og_appdata"
source ~/.zshrc 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
log "=== RFT Rollout: $NUM_ROLLOUTS trials ==="

RFT_DATA_DIR="$PLUTO/rft_data/task1_rollouts"
mkdir -p "$RFT_DATA_DIR" "$PLUTO/logs"

# 1. 啟動 policy server
log "啟動 policy server..."
CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
  uv run "$WORK/scripts/serve_b1k.py" \
    --task-checkpoint-mapping "$WORK/task_checkpoint_mapping.json" \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir "$CKPT_DIR" &
SERVER_PID=$!
sleep 30

# 2. 跑 rollouts（帶隨機擾動）
log "執行 $NUM_ROLLOUTS rollouts with perturbations..."
OMNIGIBSON_DATA_PATH="$OG_DATA" OMNIGIBSON_APPDATA_PATH="$OMNIGIBSON_APPDATA_PATH" \
OMNIGIBSON_GPU_ID=0 \
  python "$WORK/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py" \
    log_path="$RFT_DATA_DIR" \
    policy=websocket \
    task.name=picking_up_trash \
    model.host=localhost \
    eval_instance_ids="[0,1,2,3,4,5,6,7,8,9]" \
    num_trials_per_instance=$((NUM_ROLLOUTS / 10)) \
    save_trajectories=true \
    use_perturbations=true \
    perturbation_scale=0.15 \
    2>&1 | tee "$PLUTO/logs/rft_rollout.log"

kill $SERVER_PID 2>/dev/null || true

# 3. 篩選成功的 trajectory
log "篩選成功的 trajectories..."
python - << 'PY'
import json, os, pathlib

rollout_dir = pathlib.Path("/media/Pluto/Shawn/NTHU_Course_1142/b1k/rft_data/task1_rollouts")
success_dir = rollout_dir / "success_only"
success_dir.mkdir(exist_ok=True)

total, kept = 0, 0
for f in sorted(rollout_dir.glob("*/episode_*.json")):
    total += 1
    with open(f) as fp:
        data = json.load(fp)
    # BDDL 成功 + 在合理時間內完成（不是超時的）
    if data.get("success", False) and data.get("length", 99999) < 15000:
        import shutil
        shutil.copy(f, success_dir / f.name)
        kept += 1

print(f"篩選結果: {kept}/{total} trajectories 保留")
print(f"成功率: {kept/total*100:.1f}%")
PY

log "=== RFT 資料篩選完成，存於 $RFT_DATA_DIR/success_only/ ==="
log "下一步: bash run_rft_finetune.sh"
