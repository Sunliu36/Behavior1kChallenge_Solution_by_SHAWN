#!/bin/bash
# 評估 6 個目標任務
# 用法: bash run_all_evals.sh [output_prefix] [n_instances]
# output_prefix: 輸出目錄前綴 (default: baseline)
# n_instances: 每個任務評估幾個 instance (default: 2, max: 20)

PLUTO="/media/Pluto/Shawn/NTHU_Course_1142/b1k"
DATA_PATH="/media/public_dataset2/behavior-1k/omnigibson_data"
PYTHON="$PLUTO/eval_venv_py310/bin/python"
EVAL_SCRIPT="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution/BEHAVIOR-1K/OmniGibson/omnigibson/learning/eval.py"

PREFIX="${1:-baseline}"
N_INST="${2:-2}"   # number of instances: eval_instance_ids=[0..N-1]

# Build instance id list like [0,1] or [0,1,2]
INST_LIST="["
for i in $(seq 0 $((N_INST - 1))); do
    [ $i -gt 0 ] && INST_LIST="${INST_LIST},"
    INST_LIST="${INST_LIST}${i}"
done
INST_LIST="${INST_LIST}]"

# 按照 timeout 長度排序（短的優先），確保能在截止前看到更多結果
TASKS=(
    "picking_up_trash"        # task 1, timeout ~10535 steps
    "tidying_bedroom"         # task 18, timeout ~22074 steps
    "sorting_household_items" # task 27, timeout ~31615 steps
    "picking_up_toys"         # task 7, timeout ~37781 steps
    "collecting_childrens_toys" # task 21, timeout ~38372 steps
    "clean_up_your_desk"      # task 29, timeout ~42857 steps
)

echo "[$(date '+%H:%M:%S')] 開始評估 6 個目標任務，instance_ids=${INST_LIST}"
echo "[$(date '+%H:%M:%S')] 輸出前綴: ${PREFIX}"

for TASK in "${TASKS[@]}"; do
    OUT_DIR="$PLUTO/outputs/${PREFIX}_${TASK}"
    LOG="$PLUTO/logs/${PREFIX}_${TASK}.log"
    echo ""
    echo "[$(date '+%H:%M:%S')] === 評估 ${TASK} ==="
    echo "[$(date '+%H:%M:%S')] 輸出: ${OUT_DIR}"

    OMNI_KIT_ACCEPT_EULA=YES \
    OMNIGIBSON_DATA_PATH="$DATA_PATH" \
    OMNIGIBSON_APPDATA_PATH="$PLUTO/outputs/og_appdata" \
    OG_APPDATA_DIR="$PLUTO/outputs/og_appdata" \
    OMNIGIBSON_GPU_ID=1 \
    OMNIGIBSON_HEADLESS=1 \
    "$PYTHON" "$EVAL_SCRIPT" \
        log_path="$OUT_DIR" \
        policy=websocket \
        task.name="$TASK" \
        model.host=localhost \
        eval_instance_ids="$INST_LIST" \
        +num_eval_episodes=1 \
        2>&1 | tee "$LOG"

    echo "[$(date '+%H:%M:%S')] ${TASK} 評估完成"
    # 印出 q_score
    python3 -c "
import json, glob
files = glob.glob('${OUT_DIR}/metrics/*.json')
for f in sorted(files):
    data = json.load(open(f))
    qs = data.get('q_score', {}).get('final', 'N/A')
    print(f'  {f.split(\"/\")[-1]}: q_score={qs}')
" 2>/dev/null || echo "  (no metrics yet)"
done

echo ""
echo "[$(date '+%H:%M:%S')] === 所有任務評估完成 ==="
echo "結果摘要:"
for TASK in "${TASKS[@]}"; do
    OUT_DIR="$PLUTO/outputs/${PREFIX}_${TASK}"
    echo "  ${TASK}:"
    python3 -c "
import json, glob
files = glob.glob('${OUT_DIR}/metrics/*.json')
total_qs = 0
n = 0
for f in sorted(files):
    data = json.load(open(f))
    qs = data.get('q_score', {}).get('final', 0)
    if qs != 'N/A':
        total_qs += float(qs)
        n += 1
if n > 0:
    print(f'    avg q_score = {total_qs/n:.4f} ({n} instances)')
else:
    print('    no results')
" 2>/dev/null
done
