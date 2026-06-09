#!/bin/bash
# 等待 baseline_task18 eval 完成後自動啟動 fine-tuning
# 背景執行: nohup bash auto_start_finetune.sh > /tmp/auto_ft.log 2>&1 &

PLUTO="/media/Pluto/Shawn/NTHU_Course_1142/b1k"
LOG18="$PLUTO/logs/baseline_task18_v3.log"
EVAL_PID=101468

echo "[$(date '+%H:%M:%S')] 等待 task 18 eval 完成 (PID=$EVAL_PID)..."

# Wait for eval process to finish
while ps -p $EVAL_PID > /dev/null 2>&1; do
    sleep 60
    STEP=$(grep "Current step" "$LOG18" | tail -1 | grep -oP 'step: \K\d+' || echo "?")
    echo "[$(date '+%H:%M:%S')] 評估中... 最新 step=$STEP"
done

echo "[$(date '+%H:%M:%S')] Eval 進程結束！"

# Check metrics
python3 -c "
import json, glob
for f in sorted(glob.glob('$PLUTO/outputs/baseline_task18/metrics/*.json')):
    data = json.load(open(f))
    qs = data.get('q_score', {}).get('final', 'N/A')
    print(f'{f.split(\"/\")[-1]}: q_score={qs}')
" 2>/dev/null || echo "無法讀取 metrics"

echo "[$(date '+%H:%M:%S')] 停止 policy server..."
pkill -f "serve_b1k.py" 2>/dev/null && echo "Policy server 已停止" || echo "Policy server 已不存在"
sleep 5

echo "[$(date '+%H:%M:%S')] 啟動 fine-tuning: ft_ckpt2_tasks1_7_18_21"
bash /media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/script/launch_finetune.sh ft_ckpt2_tasks1_7_18_21

echo "[$(date '+%H:%M:%S')] Fine-tuning 已在背景啟動。"
echo "監控: tail -f $PLUTO/logs/finetune_ft_ckpt2_tasks1_7_18_21.log"
