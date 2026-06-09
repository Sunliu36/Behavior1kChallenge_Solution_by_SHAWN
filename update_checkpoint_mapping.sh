#!/bin/bash
# Fine-tune 完成後更新 task_checkpoint_mapping.json
# 用法: bash update_checkpoint_mapping.sh [exp_name] [step]
# exp_name: ft_ckpt2 (default)
# step: checkpoint step number (default: auto-detect latest)

PLUTO="/media/Pluto/Shawn/NTHU_Course_1142/b1k"
WORK="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution"
MAPPING="$WORK/task_checkpoint_mapping.json"

EXP="${1:-ft_ckpt2}"
STEP="${2:-}"

# Find checkpoint directory
CKPT_BASE="$PLUTO/outputs/checkpoints"
# ft_ckpt2_tasks1_7_18_21 saves to ft_ckpt2 subdir
# checkpoint_dir = checkpoint_base_dir / name / exp_name
# For ft_ckpt2_tasks1_7_18_21: name="ft_ckpt2_tasks1_7_18_21", exp_name="ft_ckpt2"
CKPT_DIR="$CKPT_BASE/ft_ckpt2_tasks1_7_18_21/${EXP}"

if [ ! -d "$CKPT_DIR" ]; then
    echo "ERROR: Checkpoint directory not found: $CKPT_DIR"
    echo "Available checkpoints:"
    ls "$CKPT_BASE/" 2>/dev/null
    exit 1
fi

# Find latest step if not specified
if [ -z "$STEP" ]; then
    STEP=$(ls "$CKPT_DIR/" | grep "^[0-9]" | sort -n | tail -1)
fi

if [ -z "$STEP" ]; then
    echo "ERROR: No checkpoint steps found in $CKPT_DIR"
    ls "$CKPT_DIR/"
    exit 1
fi

NEW_CKPT_PATH="$CKPT_DIR/$STEP"
echo "[$(date '+%H:%M:%S')] 更新 checkpoint_2 → $NEW_CKPT_PATH"

# Backup original
cp "$MAPPING" "${MAPPING}.bak"

# Update mapping with Python
python3 -c "
import json

mapping_file = '${MAPPING}'
new_path = '${NEW_CKPT_PATH}'

with open(mapping_file) as f:
    data = json.load(f)

# Update checkpoint_2 path
old_path = data['checkpoints']['checkpoint_2']['path']
data['checkpoints']['checkpoint_2']['path'] = new_path
print(f'checkpoint_2: {old_path} -> {new_path}')

# Move tasks 27 and 29 from their original checkpoints to checkpoint_2
# (since we fine-tuned checkpoint_2 on those tasks too)
for ckpt_name in ['checkpoint_1', 'checkpoint_3']:
    if ckpt_name in data['checkpoints']:
        tasks = data['checkpoints'][ckpt_name]['tasks']
        to_move = [t for t in tasks if t in [27, 29]]
        if to_move:
            for t in to_move:
                tasks.remove(t)
                if t not in data['checkpoints']['checkpoint_2']['tasks']:
                    data['checkpoints']['checkpoint_2']['tasks'].append(t)
            print(f'Moved tasks {to_move} from {ckpt_name} to checkpoint_2')

with open(mapping_file, 'w') as f:
    json.dump(data, f, indent=2)
print('Mapping updated successfully.')
"

echo "[$(date '+%H:%M:%S')] 重啟 policy server..."
# Kill existing policy server
POLICY_PID=$(ps aux | grep "serve_b1k.py" | grep -v grep | awk '{print $2}')
if [ -n "$POLICY_PID" ]; then
    echo "Killing policy server PID $POLICY_PID"
    kill "$POLICY_PID"
    sleep 3
fi

# Restart policy server (same approach as existing server)
NAS="/media/ML_2025/shawn/b1k"
LOG="$PLUTO/logs/policy_server_ft.log"

cd "$WORK"
CUDA_VISIBLE_DEVICES=2 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
nohup "$NAS/venv/bin/python" -u "$WORK/scripts/serve_b1k.py" \
    --task-checkpoint-mapping "$MAPPING" \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir "$NEW_CKPT_PATH" \
    > "$LOG" 2>&1 &

echo "Policy server PID: $!"
echo "Log: $LOG"
echo "等待 policy server 啟動 (30s)..."
sleep 30
tail -5 "$LOG"
echo "[$(date '+%H:%M:%S')] Policy server 應已啟動"
