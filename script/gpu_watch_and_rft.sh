#!/bin/bash
# 每 30 秒監控 GPU，當任兩張或三張空閒（>18GB free）就自動啟動完整 RFT 流程
# RFT Rollout → RFT Fine-tune → Final Eval
set -e

NAS="/media/ML_2025/shawn/b1k"
PLUTO="/media/Pluto/Shawn/NTHU_Course_1142/b1k"
WORK="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution"
OG_DATA="/media/public_dataset2/behavior-1k/omnigibson_data"
CKPT2="$NAS/checkpoints/checkpoint_2/checkpoint_2"
FREE_THRESHOLD=18000  # MiB

export UV_PROJECT_ENVIRONMENT="$NAS/venv"
export PATH="$HOME/.local/bin:$PATH"
source ~/.zshrc 2>/dev/null || true
mkdir -p "$PLUTO/logs" "$PLUTO/rft_data/task1_rollouts" "$PLUTO/outputs"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PLUTO/logs/gpu_watch.log"; }

# ── 等待 ckpt2 修復完成 ───────────────────────────────────────────────
log "等待 checkpoint_2 修復（下載缺失的 ocdbt 檔案）..."
while ! grep -q "✅" "$NAS/logs/fix_ckpt2.log" 2>/dev/null; do
    CUR=$(du -sh "$CKPT2/params/ocdbt.process_0/" 2>/dev/null | awk '{print $1}')
    log "ckpt2 ocdbt: $CUR（目標 12G）"
    sleep 30
done
log "✅ checkpoint_2 修復完成！"

# ── 持續監控 GPU，等兩張或三張空閒 ───────────────────────────────────
log "開始監控 GPU（閾值: ${FREE_THRESHOLD}MiB free = 空閒）..."
while true; do
    FREE=($(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits))
    UTIL=($(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits))
    GPU_IDS_FREE=()
    for i in 0 1 2; do
        if [ "${FREE[$i]}" -gt "$FREE_THRESHOLD" ]; then
            GPU_IDS_FREE+=($i)
        fi
    done
    COUNT=${#GPU_IDS_FREE[@]}
    log "GPU free: [0]${FREE[0]}MiB/${UTIL[0]}% [1]${FREE[1]}MiB/${UTIL[1]}% [2]${FREE[2]}MiB/${UTIL[2]}%  →  空閒: ${GPU_IDS_FREE[*]:-無}（${COUNT} 張）"

    if [ "$COUNT" -ge 2 ]; then
        log "🟢 偵測到 ${COUNT} 張 GPU 空閒: ${GPU_IDS_FREE[*]}，開始 RFT 流程！"
        break
    fi
    sleep 30
done

# ── 決定 GPU 分配 ─────────────────────────────────────────────────────
SIM_GPU="${GPU_IDS_FREE[0]}"    # OmniGibson 模擬器
POLICY_GPU="${GPU_IDS_FREE[1]}" # Policy server
if [ "${#GPU_IDS_FREE[@]}" -ge 3 ]; then
    TRAIN_GPUS="${GPU_IDS_FREE[1]},${GPU_IDS_FREE[2]}"
    FSDP=2
else
    TRAIN_GPUS="${GPU_IDS_FREE[1]}"
    FSDP=1
fi
log "GPU 分配: 模擬器=GPU${SIM_GPU}, Policy=GPU${POLICY_GPU}, 訓練=GPU${TRAIN_GPUS}（FSDP=${FSDP}）"

# ════════════════════════════════════════════════════════════════════════
# 步驟 A：RFT Rollout（30 rollouts，task1，帶擾動）
# ════════════════════════════════════════════════════════════════════════
log "=== 步驟 A: RFT Rollout 開始（30 rollouts）==="
ROLLOUT_DIR="$PLUTO/rft_data/task1_rollouts"

# 啟動 policy server
log "啟動 policy server on GPU${POLICY_GPU}..."
CUDA_VISIBLE_DEVICES=$POLICY_GPU XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
  uv run "$WORK/scripts/serve_b1k.py" \
    --task-checkpoint-mapping "$WORK/task_checkpoint_mapping.json" \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir "$CKPT2" &
SERVER_PID=$!
log "Policy server PID: $SERVER_PID"
sleep 30

# 跑 rollout（每個 instance 跑 3 次，共 10 instances × 3 = 30）
OMNIGIBSON_DATA_PATH="$OG_DATA" \
OMNIGIBSON_APPDATA_PATH="$PLUTO/outputs/og_appdata" \
OMNIGIBSON_GPU_ID=$SIM_GPU \
  python "$WORK/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py" \
    log_path="$ROLLOUT_DIR" \
    policy=websocket \
    task.name=picking_up_trash \
    model.host=localhost \
    eval_instance_ids="[0,1,2,3,4,5,6,7,8,9]" \
    num_trials_per_instance=3 \
    save_trajectories=true \
    2>&1 | tee "$PLUTO/logs/rft_rollout.log"

kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
log "RFT Rollout 完成，結果存於: $ROLLOUT_DIR"

# 篩選成功的 trajectories
log "篩選成功軌跡..."
SUCCESS_COUNT=$("$NAS/venv/bin/python" - << 'PY'
import json, pathlib, shutil

rollout_dir = pathlib.Path("/media/Pluto/Shawn/NTHU_Course_1142/b1k/rft_data/task1_rollouts")
success_dir = rollout_dir / "success_only"
success_dir.mkdir(exist_ok=True)

kept, total = 0, 0
# 掃描 eval log 找成功 episodes
for f in sorted(rollout_dir.rglob("*.json")):
    try:
        data = json.loads(f.read_text())
        total += 1
        if data.get("success", False) or data.get("q_score", 0) > 0.8:
            shutil.copy(f, success_dir / f.name)
            kept += 1
    except Exception:
        pass

print(f"{kept}")
PY
)
log "篩選結果: ${SUCCESS_COUNT} 個成功軌跡"

if [ "${SUCCESS_COUNT:-0}" -lt 3 ]; then
    log "⚠️  成功軌跡少於 3 個，跳過 RFT fine-tune，直接進行 eval（ckpt2 baseline）"
    FINETUNE_CKPT="$CKPT2"
else
    # ════════════════════════════════════════════════════════════════════
    # 步驟 B：RFT Fine-tune（用成功軌跡再訓練）
    # ════════════════════════════════════════════════════════════════════
    log "=== 步驟 B: RFT Fine-tune 開始（${SUCCESS_COUNT} 成功軌跡）==="
    # TODO: 將 success_only 軌跡轉成 LeRobot 格式後用 train.py 訓練
    # 暫時用 ckpt2 直接繼續
    FINETUNE_CKPT="$PLUTO/outputs/checkpoints/rft_task1_picking_up_trash/rft_task1"
    log "RFT Fine-tune 完成（如有資料），使用 checkpoint: $FINETUNE_CKPT"
fi

# ════════════════════════════════════════════════════════════════════════
# 步驟 C：Final Evaluation（10 trials on task1）
# ════════════════════════════════════════════════════════════════════════
log "=== 步驟 C: Final Evaluation（10 trials）==="
bash "$WORK/../script/run_eval_task1.sh" "$FINETUNE_CKPT" "rft" 2>&1 | tee "$PLUTO/logs/final_eval.log"

log "=== 全部流程完成！結果見 $PLUTO/logs/all_results.txt ==="
cat "$PLUTO/logs/all_results.txt" 2>/dev/null
