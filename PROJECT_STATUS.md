# BEHAVIOR-1K Challenge — Project Status Handoff

**Last updated**: 2026-06-06 ~20:55  
**Current server**: `QAQ` (Linux, `/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/`)  
**User**: shawn (`huchch`)

---

## 1. 專案來源與目標

### 背景
- **競賽**: [2025 BEHAVIOR Challenge (NeurIPS 2025)](https://behavior.stanford.edu/challenge/)
- **基礎方案**: 使用一個已知的 winning solution repo（26% success rate），基於 [Pi0.5 VLA model](https://www.physicalintelligence.company/blog/pi0)
- **Repo**: `behavior-1k-solution/` (fork of [IliaLarchenko's solution](https://huggingface.co/IliaLarchenko/behavior_submission))
- **模型架構**: PiBehavior（Pi0 + FAST tokenization + subtask state prediction + KV transform）

### 目標
在 6 個目標任務上最大化 **q_score**：

| Challenge Task ID | Task Name | Timeout (sim steps) | Baseline q_score |
|---|---|---|---|
| 1 | `picking_up_trash` | 10,535 | 0.0 ★ |
| 7 | `picking_up_toys` | 37,781 | 未測（推估 0.0）|
| 18 | `tidying_bedroom` | 22,074 | 0.0 ★ |
| 21 | `collecting_childrens_toys` | 38,372 | 未測（推估 0.0）|
| 27 | `sorting_household_items` | 31,615 | 未測；task 27 只有 head camera，136/200 episodes 有 video |
| 29 | `clean_up_your_desk` | 42,857 | 未測；**完全無 training data**，無法 fine-tune |

> ★ **Baseline 實驗說明**（2026-06-04，fine-tune 之前）：
> - 使用原始 IliaLarchenko **checkpoint_2**（未 fine-tune），透過 `task_checkpoint_mapping.json` 的 multi-checkpoint 模式對應
> - `picking_up_trash`：跑了 **10 個 instances**，結果存於 `$PLUTO/outputs/final_eval/metrics/`，全部 q=0.0（每次跑滿 10535 steps timeout）
> - `tidying_bedroom`：跑了 **2 個 instances**，結果存於 `$PLUTO/outputs/baseline_task18/`，全部 q=0.0（每次跑滿 22074 steps timeout）
> - 結論：原版 checkpoint 對這 6 個 target tasks 直接 inference 全部失敗，因此需要 fine-tune

---

## 2. 目錄結構

```
/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/
├── behavior-1k-solution/          # 主要 code repo（MODIFIED）
│   ├── scripts/
│   │   ├── train.py               # ★ MODIFIED：全部 params 轉 bfloat16 + format fix
│   │   └── serve_b1k.py           # Policy server 主程式
│   ├── src/b1k/
│   │   ├── training/
│   │   │   ├── config.py          # ★ MODIFIED：新增 filter_cameras/filter_tasks/episodes_index
│   │   │   └── data_loader.py     # ★ MODIFIED：新增 cameras/task_names 參數
│   │   └── policies/
│   │       └── b1k_policy.py      # ★ MODIFIED：wrist camera 缺失時 zero-fill
│   └── task_checkpoint_mapping.json  # 任務→checkpoint 對應表
├── src/
│   └── collect_rft_data.py        # rollout → LeRobot 收集器
├── script/                        # 所有 orchestration / eval shell scripts
│   ├── launch_finetune.sh         # GPU2 啟動 fine-tune
│   ├── launch_finetune_gpu1.sh    # GPU1 啟動 fine-tune（備用）
│   ├── auto_post_finetune.sh      # FT 完成後自動更新 ckpt + eval
│   ├── update_checkpoint_mapping.sh # 更新 task_checkpoint_mapping.json + 重啟 policy server
│   ├── run_all_evals.sh           # 跑全部 6 tasks eval
│   └── run_baseline_eval.sh       # 跑 baseline eval
└── PROJECT_STATUS.md              # 本文件

/media/ML_2025/shawn/b1k/         # NAS 儲存（code/models/data）
├── venv/                          # Python 虛擬環境（Python 3.11）
├── checkpoints/
│   ├── checkpoint_1/              # 用於 task 29 等
│   ├── checkpoint_2/              # 用於 task 1,7,18,21（★ 正在 fine-tune）
│   ├── checkpoint_3/              # 用於 task 27
│   └── checkpoint_4/              # 其他 tasks
├── data/behavior_224_rgb/         # Training data（LeRobot 格式）
└── outputs/assets/                # Norm stats（預先計算好）

/media/Pluto/Shawn/NTHU_Course_1142/b1k/   # Pluto 儲存（輸出/logs/checkpoints）
├── logs/
│   ├── finetune_ft_ckpt2_tasks1_7_18_21.log  # ★ 當前訓練 log（GPU2）
│   ├── finetune_pid.txt                       # 當前訓練 PID
│   ├── auto_post_ft.log                       # 後處理 watcher log
│   └── final_eval_all_tasks.log               # Eval 完成後在此
└── outputs/
    ├── checkpoints/
    │   └── ft_ckpt2_tasks1_7_18_21/ft_ckpt2/ # ★ FT 輸出 checkpoint（每 500 steps 存一次）
    │       ├── 2000/
    │       ├── 4000/
    │       └── 4500/               # 最新（截至文件時）
    └── final_eval/metrics/         # Baseline eval 結果（都是 0.0）
```

---

## 3. 現在的狀態（2026-06-06 20:55）

### GPU 使用情況
| GPU | 使用量 | 用途 |
|---|---|---|
| GPU 0 | 7190 MiB | **別人的進程**（PID 234974，勿動） |
| GPU 1 | 648 MiB | Policy server（idle，等 FT 完成後 eval 使用）|
| GPU 2 | 18903 MiB | **ft_ckpt2_tasks1_7_18_21 fine-tuning 中** |

### Fine-tuning 進度
- **Config**: `ft_ckpt2_tasks1_7_18_21`
- **PID**: 3241160（可用 `ps -p 3241160` 確認）
- **進度**（截至 20:50）：4960/10000 steps，速度 1.4 s/it
- **預計完成**：~22:44（剩約 1h54m）
- **Log**: `/media/Pluto/Shawn/NTHU_Course_1142/b1k/logs/finetune_ft_ckpt2_tasks1_7_18_21.log`

### 自動後處理 Watcher
- **PID**: 448112（`auto_post_finetune.sh`，背景運行中）
- FT 完成後自動執行：
  1. `bash script/update_checkpoint_mapping.sh ft_ckpt2` → 更新 mapping + 重啟 policy server
  2. `bash script/run_all_evals.sh ft_eval 2` → 跑 6 tasks × 2 instances
  3. 結果存到 `$PLUTO/outputs/ft_eval_*/metrics/`

---

## 4. 關鍵修改（相對於原始 repo）

### 4.1 `scripts/train.py`（核心 OOM 修復）
```python
# 原本（只轉 frozen LLM params）：
params = nnx_utils.state_map(params, config.freeze_filter,
    lambda p: p.replace(p.value.astype(jnp.bfloat16)))

# 修改後（★ ALL params 轉 bfloat16，GPU peak 從 23.75 GB → 17.56 GB）：
params = params.map(lambda k, v: v.replace(v.value.astype(jnp.bfloat16)))
```

另外修復 metrics 格式化 bug（train step 437 附近）：
```python
# 原本（會對 str 使用 :.4f 而 crash）：
info_str = ", ".join(f"{k}={v:.4f}" for k, v in main_metrics.items())

# 修改後：
parts = []
for k, v in main_metrics.items():
    try:
        parts.append(f"{k}={float(v):.4f}")
    except (TypeError, ValueError):
        parts.append(f"{k}={v}")
info_str = ", ".join(parts)
```

### 4.2 `src/b1k/training/config.py`
- `LeRobotB1KDataConfig` 新增欄位 `filter_task_names`, `filter_cameras`
- `_ft_data()` 新增 `filter_tasks`, `filter_cameras`, `episodes_index` 參數
- `ft_ckpt2_tasks1_7_18_21` config：`batch_size=8`, `num_flow_samples=2`, head-only camera, 4 tasks
- `ft_ckpt3_task27` config：`filter_cameras=["head"]`, `episodes_index=list(range(136))`（只有 136 個有 video）

### 4.3 `src/b1k/training/data_loader.py`
- `create_behavior_dataset()` 接受 `task_names` 和 `cameras` 參數
- `create_behavior_data_loader()` 從 `config.data` 取出 filter 設定

### 4.4 `src/b1k/policies/b1k_policy.py`
- `B1kInputs.__call__` 對缺少 wrist camera 的資料做 graceful handling（zero-fill + `image_mask=False`）

---

## 5. 遇到的坑 & 解法

| 問題 | 原因 | 解法 |
|---|---|---|
| OOM（5.45 GiB 無法分配） | 模型 float32 訓練需要 23.75 GB，超過 24 GB GPU | 將所有 params 轉 bfloat16，峰值降至 17.56 GB |
| `ValueError: format code 'f' for str` | metrics dict 中有 numpy str 型別值 | try/except float() 轉換 |
| `AssertionError: Missing file episode_*.mp4` | task 27/29 的 dataset 缺少部分 video | task 27：用 `episodes_index=list(range(136))` 跳過缺失的 64 個；task 29：完全無資料，略過 |
| `FileExistsError: checkpoint dir already exists` | 前次崩潰留下殘留目錄 | 每次啟動前 `rm -rf $CKPT_DIR` |
| HF cache `OSError: No space left on device` | 根磁碟（11 GB）被 HF datasets 撐滿 | 將 `HF_HOME` + `HF_DATASETS_CACHE` 改到 Pluto |
| `KeyError: 'observation/wrist_image_left'` | head-only camera 訓練時 B1kInputs 硬寫 key | 加上 `has_left/has_right` 判斷 |
| `episodes` 參數格式錯誤 | `BehaviorLeRobotDataset.episodes` 是各 task 內的 0-based positional index，不是 episode_index 值 | 改用 `list(range(N_valid_episodes))` |

---

## 6. Fine-tuning Config 細節

### 當前執行中的 Config（`ft_ckpt2_tasks1_7_18_21`）
```python
TrainConfig(
    name="ft_ckpt2_tasks1_7_18_21",
    exp_name="ft_ckpt2",
    model=_ft_model(),             # PiBehavior, use_fast=True, freeze_vision=True
    data=_ft_data(
        filter_tasks=["picking_up_trash", "picking_up_toys",
                      "tidying_bedroom", "collecting_childrens_toys"],
        filter_cameras=["head"],   # 只用 head camera（wrist videos 缺失 or 省記憶體）
    ),
    lr_schedule=CosineDecay(warmup=200, peak=5e-5, decay=10000, end=5e-6),
    batch_size=8,
    num_flow_samples=2,
    freeze_filter=PathRegex(".*PaliGemma.llm.*"),  # 凍結 LLM
    weight_loader=PiBehaviorWeightLoader(".../checkpoint_2/params"),
    num_train_steps=10000,
    save_interval=500,
    keep_period=2000,
)
```

### 記憶體分配（bfloat16 後）
- GPU2 peak: **17.56 GiB** / 24.56 GiB
- 比 float32 訓練（23.75 GiB）少 **6.2 GiB**

---

## 7. 接手後要做的事

### 情境一：Fine-tuning 仍在跑
```bash
# 確認訓練還活著
ps -p $(cat /media/Pluto/Shawn/NTHU_Course_1142/b1k/logs/finetune_pid.txt)

# 看訓練進度
tail -5 /media/Pluto/Shawn/NTHU_Course_1142/b1k/logs/finetune_ft_ckpt2_tasks1_7_18_21.log

# 確認 auto_post_finetune.sh 在跑
ps aux | grep auto_post_finetune.sh | grep -v grep
```

### 情境二：Fine-tuning 崩潰（需要重啟）
```bash
PLUTO="/media/Pluto/Shawn/NTHU_Course_1142/b1k"
# 清除舊 checkpoint dir
rm -rf "$PLUTO/outputs/checkpoints/ft_ckpt2_tasks1_7_18_21"
rm -f "$PLUTO/logs/finetune_pid.txt"

# 重啟
cd /media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd
bash script/launch_finetune.sh ft_ckpt2_tasks1_7_18_21

# 重啟 watcher
nohup bash script/auto_post_finetune.sh > "$PLUTO/logs/auto_post_ft.log" 2>&1 &
```

### 情境三：Fine-tuning 已完成，需要手動跑 eval
```bash
WORK="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd"
# 更新 checkpoint mapping + 重啟 policy server
bash "$WORK/script/update_checkpoint_mapping.sh" ft_ckpt2

# 等 policy server 啟動（約 60s）
sleep 60

# 跑全部 6 tasks eval（n_instances=2）
bash "$WORK/script/run_all_evals.sh" ft_eval 2

# 查看結果
python3 -c "
import json, glob, os
for f in sorted(glob.glob('/media/Pluto/Shawn/NTHU_Course_1142/b1k/outputs/ft_eval*/metrics/*.json')):
    d = json.load(open(f))
    print(os.path.basename(f), d.get('q_score',{}).get('final','?'))
"
```

### 情境四：換 server 要做什麼
1. **掛載磁碟**: `/media/ML_2025/shawn/b1k`（NAS）和 `/media/Pluto/Shawn/NTHU_Course_1142/b1k`（Pluto）
2. **Python 環境**: `/media/ML_2025/shawn/b1k/venv`（Python 3.11，JAX/Flax/OmniGibson 都在這）
3. **Eval 環境**: `/media/Pluto/Shawn/NTHU_Course_1142/b1k/eval_venv_py310`（Python 3.10，OmniGibson eval 用）
4. **GPU 限制**: 只用 GPU 1 和 GPU 2（GPU 0 是別人的）
5. **HF cache**: 設到 Pluto（根磁碟 100% 滿）
6. **重要的 env vars**（已寫在 `script/launch_finetune.sh`）：
   ```bash
   export CUDA_VISIBLE_DEVICES=2
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
   export HF_HOME="/media/Pluto/Shawn/NTHU_Course_1142/b1k/hf_cache"
   export HF_DATASETS_CACHE="$HF_HOME/datasets"
   ```

---

## 8. Policy Server

Policy server 負責接收 OmniGibson 的 obs，回傳 actions。

```bash
# 查看 policy server 是否在跑
ps aux | grep "serve_b1k.py" | grep -v grep

# 手動啟動（baseline checkpoint_2）
NAS="/media/ML_2025/shawn/b1k"
WORK="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution"
PLUTO="/media/Pluto/Shawn/NTHU_Course_1142/b1k"

CUDA_VISIBLE_DEVICES=1 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
nohup "$NAS/venv/bin/python" -u "$WORK/scripts/serve_b1k.py" \
    --task-checkpoint-mapping "$WORK/task_checkpoint_mapping.json" \
    policy:checkpoint \
    --policy.config pi_behavior_b1k_fast \
    --policy.dir "$NAS/checkpoints/checkpoint_2/checkpoint_2" \
    > "$PLUTO/logs/policy_server.log" 2>&1 &

# 查看 server log
tail -f /media/Pluto/Shawn/NTHU_Course_1142/b1k/logs/policy_server_ft.log
```

`task_checkpoint_mapping.json` 定義了哪個 task ID 用哪個 checkpoint：
- Task IDs 1,7,18,21 → `checkpoint_2`（fine-tune 後會更新到 ft checkpoint）
- Task ID 27 → `checkpoint_3`
- Task ID 29 → `checkpoint_1`（無訓練資料，用原始 baseline）

---

## 9. 時間估算（截至 2026-06-06 20:55）

| 階段 | 預計時間 | 說明 |
|---|---|---|
| FT 完成 | ~22:44 | 剩 1h49m，1.4 s/step |
| Checkpoint update + server restart | ~10 min | `update_checkpoint_mapping.sh` |
| 6 tasks × 2 instances eval | ~5-6 小時 | 依各 task timeout（最長 task 29 = 42857 steps ≈ 24 min/trial）|

每個 task eval 的時間（最壞情況，robot 失敗跑滿 timeout）：
- Task 1（picking_up_trash）: 351 秒 ≈ 6 分鐘
- Task 18（tidying_bedroom）: ~735 秒 ≈ 12 分鐘
- Task 27（sorting_household_items）: ~1054 秒 ≈ 18 分鐘
- Task 7 / 21 / 29: 更長

---

## 10. 有用的監控指令

```bash
# 全局狀態一覽
PLUTO="/media/Pluto/Shawn/NTHU_Course_1142/b1k"

# GPU 狀態
nvidia-smi --query-gpu=index,memory.used,memory.total,name --format=csv

# Fine-tuning 進度
grep "rate:" "$PLUTO/logs/finetune_ft_ckpt2_tasks1_7_18_21.log" | tail -1

# 已存的 checkpoints
ls "$PLUTO/outputs/checkpoints/ft_ckpt2_tasks1_7_18_21/ft_ckpt2/" | sort -n

# Eval 結果
ls "$PLUTO/outputs/" | grep "ft_eval\|baseline"
for f in "$PLUTO/outputs/ft_eval"*/metrics/*.json; do
    echo "$f: $(python3 -c "import json; d=json.load(open('$f')); print(d.get('q_score',{}).get('final','?'))")"
done

# Process 狀態
ps aux | grep -E "train.py|serve_b1k|auto_post|eval.py" | grep -v grep
```

---

## 11. 已知限制

1. **Task 29（clean_up_your_desk）**: 完全沒有 training data，只能用 baseline checkpoint_1，預期 q_score=0.0
2. **Task 27（sorting_household_items）**: 只有 head camera 資料（136 episodes），fine-tune 品質可能有限
3. **Wrist cameras 未使用**: 為了節省 GPU 記憶體，目前 fine-tune 只用 head camera；inference 時 wrist images 由 zero tensor 填充（`image_mask=False`）
4. **Bfloat16 訓練**: 全程 bfloat16（非標準的 mixed precision），可能有輕微數值精度損失，但實測可收斂
5. **GPU 0 不能用**: 有另一位使用者佔用（7190 MiB），不要動

---

## 12. 重要連結

- [BEHAVIOR Challenge 官網](https://behavior.stanford.edu/challenge/)
- [IliaLarchenko solution HuggingFace](https://huggingface.co/IliaLarchenko/behavior_submission)
- [Pi0.5 Blog Post](https://www.physicalintelligence.company/blog/pi0)
- Training data: `IliaLarchenko/behavior_224_rgb`（已本地下載到 `/media/ML_2025/shawn/b1k/data/`）
