#!/bin/bash
# ══════════════════════════════════════════════════════════════════════════════
# B1K RFT 完整實驗控制器 v2（修復 gello + heredoc 問題）
# ══════════════════════════════════════════════════════════════════════════════
NAS="/media/ML_2025/shawn/b1k"
PLUTO="/media/Pluto/Shawn/NTHU_Course_1142/b1k"
WORK="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution"
OG_DATA="/media/public_dataset2/behavior-1k/omnigibson_data"
CKPT2="$NAS/checkpoints/checkpoint_2/checkpoint_2"
PYTHON_EVAL="/media/Pluto/Shawn/NTHU_Course_1142/b1k/eval_venv_py310/bin/python3"  # eval/rollout
PYTHON_TRAIN="$NAS/venv/bin/python"  # policy server + fine-tune（JAX venv）
PYTHON="$PYTHON_EVAL"
FREE_THRESHOLD=16000   # MiB

export UV_PROJECT_ENVIRONMENT="$NAS/venv"
export OMNIGIBSON_DATA_PATH="$OG_DATA"
export ISAAC_PATH="/media/Pluto/Shawn/NTHU_Course_1142/b1k/eval_venv_py310/lib/python3.10/site-packages/isaacsim"
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_APPDATA_PATH="$PLUTO/outputs/og_appdata"
source ~/.zshrc 2>/dev/null; export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$PLUTO/logs" "$PLUTO/rft_data/task1" "$PLUTO/outputs" "$OMNIGIBSON_APPDATA_PATH"

LOG="$PLUTO/logs/rft_controller.log"
RESULT="$PLUTO/logs/all_results.txt"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

gpu_status() {
    nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader,nounits | \
        awk -F',' '{printf "GPU%d:%dMiB_used/%dMiB_free/%d%%  ",$1,$2,$3,$4}'
}

get_free_gpus() {
    local threshold="${1:-$FREE_THRESHOLD}"
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
        awk -F',' -v t="$threshold" '$2>t {printf "%d ", $1}'
}

start_policy_server() {
    local GPU="$1" CKPT="$2"
    CUDA_VISIBLE_DEVICES=$GPU XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_MODE=disabled \
      "$PYTHON_TRAIN" "$WORK/scripts/serve_b1k.py" \
        --task-checkpoint-mapping "$WORK/task_checkpoint_mapping.json" \
        policy:checkpoint \
        --policy.config pi_behavior_b1k_fast \
        --policy.dir "$CKPT" >> "$PLUTO/logs/policy_server.log" 2>&1 &
    echo $!
}

run_eval() {
    # $1=SIM_GPU $2=INSTANCES $3=EPISODES_PER_INSTANCE $4=OUT_DIR
    local SIM="$1" INSTANCES="$2" EPISODES="$3" OUT="$4"
    mkdir -p "$OUT"
    OMNIGIBSON_DATA_PATH="$OG_DATA" \
    OMNIGIBSON_APPDATA_PATH="$OMNIGIBSON_APPDATA_PATH" \
    OMNIGIBSON_GPU_ID=$SIM \
      "$PYTHON" "$WORK/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py" \
        log_path="$OUT" \
        policy=websocket \
        task.name=picking_up_trash \
        model.host=localhost \
        eval_instance_ids="$INSTANCES" \
        +num_eval_episodes=$EPISODES 2>&1
}

# ════════════════════════════════════════════════════════════════════════════
log "╔═══ RFT 控制器 v2 啟動 ═══╗"

# 階段 0：確認 ckpt2 完整
log "階段 0：確認 checkpoint_2..."
while ! grep -q "✅" "$NAS/logs/fix_ckpt2.log" 2>/dev/null; do
    if ! pgrep -f "fix_ckpt2\|wget.*9e514607" > /dev/null 2>&1; then
        log "ckpt2 下載進程不見，重啟 wget..."
        FILE_HASH="9e514607f3cc05a8059d164663798840"
        DEST="$CKPT2/params/ocdbt.process_0/d/$FILE_HASH"
        URL="https://huggingface.co/IliaLarchenko/behavior_submission/resolve/main/checkpoint_2/params/ocdbt.process_0/d/$FILE_HASH"
        wget -q --header="Authorization: Bearer $HF_TOKEN" --continue -O "$DEST" "$URL" &
        WGET_PID=$!
        ( while kill -0 $WGET_PID 2>/dev/null; do sleep 15; done
          echo "[$(date '+%H:%M:%S')] ✅ checkpoint_2 修復完成！" >> "$NAS/logs/fix_ckpt2.log" ) &
    fi
    log "等 ckpt2... | $(gpu_status)"
    sleep 30
done
log "✅ checkpoint_2 完整"

# 階段 1：等 GPU 空閒
log "階段 1：等 GPU 空閒（需 ≥1 張 >${FREE_THRESHOLD}MiB free）..."
while true; do
    FREE_GPUS=($(get_free_gpus))
    log "$(gpu_status) | 空閒 GPU: [${FREE_GPUS[*]:-無}]"
    [ "${#FREE_GPUS[@]}" -ge 1 ] && break
    sleep 30
done
log "🟢 空閒 GPU: [${FREE_GPUS[*]}]"

SIM_GPU="${FREE_GPUS[0]}"
POLICY_GPU="${FREE_GPUS[0]}"
TRAIN_GPUS="${FREE_GPUS[0]}"
FSDP=1
if [ "${#FREE_GPUS[@]}" -ge 2 ]; then
    POLICY_GPU="${FREE_GPUS[1]}"
    TRAIN_GPUS="${FREE_GPUS[0]},${FREE_GPUS[1]}"
    FSDP=2
fi
BATCH=$((FSDP * 4))
log "分配: 模擬=GPU${SIM_GPU}, Policy=GPU${POLICY_GPU}, 訓練=GPU${TRAIN_GPUS}(FSDP=${FSDP},batch=${BATCH})"

# 階段 2：Pipeline Test（2 trials 驗證）
log "階段 2：Pipeline Test（2 trials，ckpt2）..."
SERVER_PID=$(start_policy_server "$POLICY_GPU" "$CKPT2")
log "Policy server PID=$SERVER_PID，等 40 秒..."
sleep 40

run_eval "$SIM_GPU" "[0,1]" 1 "$PLUTO/outputs/pipeline_test" "false" \
    2>&1 | tee "$PLUTO/logs/pipeline_test_eval.log"

TEST_LINES=$(wc -l < "$PLUTO/logs/pipeline_test_eval.log" 2>/dev/null || echo 0)
log "Pipeline test 完成（log ${TEST_LINES} 行）"

# 階段 3：RFT Rollout（80 rollouts）
log "階段 3：RFT Rollout（80 rollouts with perturbations）..."
mkdir -p "$PLUTO/rft_data/task1"
run_eval "$SIM_GPU" "[0,1,2,3,4,5,6,7,8,9]" 8 "$PLUTO/rft_data/task1" "true" \
    2>&1 | tee "$PLUTO/logs/rft_rollout.log"

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; sleep 3

# 篩選成功軌跡
log "篩選成功軌跡..."
FILTER_OUT=$("$PYTHON" /tmp/filter_rollouts.py 2>&1)
log "篩選結果: $FILTER_OUT"
SUCCESS_COUNT=$(echo "$FILTER_OUT" | grep "SUCCESS_COUNT=" | cut -d= -f2 | tr -d '[:space:]')
SUCCESS_COUNT="${SUCCESS_COUNT:-0}"
log "成功軌跡數: $SUCCESS_COUNT"

# 階段 4：RFT Fine-tune
FINETUNE_CKPT="$CKPT2"
if [ "$SUCCESS_COUNT" -ge 3 ] 2>/dev/null; then
    log "階段 4：RFT Fine-tune（${SUCCESS_COUNT} 軌跡，3000 steps）..."

    # 重新取得空閒 GPU（fine-tune 可能需要不同 GPU）
    sleep 5
    FREE_GPUS=($(get_free_gpus))
    TRAIN_GPUS="${FREE_GPUS[0]:-1}"
    FSDP=1
    [ "${#FREE_GPUS[@]}" -ge 2 ] && TRAIN_GPUS="${FREE_GPUS[0]},${FREE_GPUS[1]}" && FSDP=2
    BATCH=$((FSDP * 4))

    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    CUDA_VISIBLE_DEVICES=$TRAIN_GPUS WANDB_MODE=disabled \
      "$PYTHON_TRAIN" "$WORK/scripts/train.py" rft_task1_picking_up_trash \
        --batch_size=$BATCH --fsdp_devices=$FSDP \
        --num_train_steps=3000 --save_interval=500 \
        --keep_period=1000 --log_interval=25 --overwrite \
        2>&1 | tee "$PLUTO/logs/rft_finetune.log"

    # 找最新的 checkpoint
    FT_BASE="$PLUTO/outputs/checkpoints/rft_task1_picking_up_trash/rft_task1"
    if [ -d "$FT_BASE" ]; then
        FINETUNE_CKPT="$FT_BASE"
        log "RFT Fine-tune 完成，使用: $FINETUNE_CKPT"
    else
        log "Fine-tune checkpoint 不存在，使用 ckpt2"
    fi
else
    log "成功軌跡不足（${SUCCESS_COUNT} < 3），跳過 fine-tune，直接評估 ckpt2"
fi

# 階段 5：Final Evaluation（10 trials）
log "階段 5：Final Evaluation（10 trials）..."
sleep 5
FREE_GPUS=($(get_free_gpus))
SIM_GPU="${FREE_GPUS[0]:-0}"
POLICY_GPU="${FREE_GPUS[1]:-${FREE_GPUS[0]:-1}}"

SERVER_PID=$(start_policy_server "$POLICY_GPU" "$FINETUNE_CKPT")
log "Final eval server PID=$SERVER_PID，等 40 秒..."
sleep 40

run_eval "$SIM_GPU" "[0,1,2,3,4,5,6,7,8,9]" 1 "$PLUTO/outputs/final_eval" "false" \
    2>&1 | tee "$PLUTO/logs/final_eval.log"

kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null

SUCCESS_RATE=$(grep -oP "success_rate[:\s]+\K[\d.]+" "$PLUTO/logs/final_eval.log" | tail -1 || echo "see_log")
QSCORE=$(grep -oP "q_score[:\s]+\K[\d.]+" "$PLUTO/logs/final_eval.log" | tail -1 || echo "see_log")

log "═══════════════════════════════════════════════════"
log "🏁 實驗完成！"
log "Checkpoint: $FINETUNE_CKPT"
log "RFT rollout 成功軌跡: ${SUCCESS_COUNT}/${80}"
log "Final success_rate: $SUCCESS_RATE"
log "Final q_score: $QSCORE"
log "═══════════════════════════════════════════════════"
echo "FINAL ckpt2+RFT task1: success=${SUCCESS_RATE} q_score=${QSCORE} rft_successes=${SUCCESS_COUNT}" >> "$RESULT"
cat "$RESULT"
