#!/bin/bash
# Phase 1: 環境設定 + 所有資料下載（不需要 sudo）
# 全程在 tmux 中跑，SSH 斷線後照常執行
set -e
WORK_DIR="/media/extra_home/huchch/shawn/DESKTOP/B1K_1st_with_2nd/behavior-1k-solution"
NAS_DIR="/media/ML_2025/shawn/b1k"
LOG_DIR="$NAS_DIR/logs"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cd "$WORK_DIR"
log "=== Phase 1 開始 ==="
log "WORK_DIR: $WORK_DIR"
log "NAS_DIR:  $NAS_DIR"

# ── Step 1: 安裝 uv（不需要 sudo，裝在 ~/.local/bin）────────────────
log ">>> Step 1: 確認 uv..."
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    log "安裝 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
log "uv: $(uv --version)"

# ── Step 2: 設定 git LFS skip（加速 clone）──────────────────────────
log ">>> Step 2: 設定 git LFS skip..."
export GIT_LFS_SKIP_SMUDGE=1
git config --global filter.lfs.smudge  "git-lfs smudge --skip %f || cat" || true
git config --global filter.lfs.process "git-lfs filter-process --skip || cat" || true
git config --global lfs.fetchinclude "" || true
git config --global lfs.fetchexclude "*" || true

# ── Step 3: 確認 submodules ──────────────────────────────────────────
log ">>> Step 3: 確認 submodules..."
[ ! -d "openpi/.git" ]     && GIT_LFS_SKIP_SMUDGE=1 git submodule update --init openpi
[ ! -d "BEHAVIOR-1K/.git" ] && GIT_LFS_SKIP_SMUDGE=1 git submodule update --init BEHAVIOR-1K
log "Submodules OK"

# ── Step 4: uv sync（venv 放 NAS 節省本地空間）──────────────────────
log ">>> Step 4: uv sync (venv → $NAS_DIR/venv)..."
export UV_PROJECT_ENVIRONMENT="$NAS_DIR/venv"
export UV_LINK_MODE=copy
uv sync --extra dev
log "uv sync 完成"

# ── Step 5: 安裝 BEHAVIOR-1K 子套件────────────────────────────────
# uv 預設不安裝 pip，用 python -m pip 安裝（先確保 pip 在 venv 裡）
log ">>> Step 5: 安裝 bddl + OmniGibson..."
VENV_PYTHON="$NAS_DIR/venv/bin/python"
"$VENV_PYTHON" -m ensurepip --upgrade 2>/dev/null || true
"$VENV_PYTHON" -m pip install --upgrade pip --quiet
cd "$WORK_DIR/BEHAVIOR-1K"
"$VENV_PYTHON" -m pip install -e bddl
"$VENV_PYTHON" -m pip install -e "OmniGibson[eval]"
cd "$WORK_DIR"
log "BEHAVIOR-1K 依賴安裝完成"

# ── Step 6: HuggingFace 登入────────────────────────────────────────
log ">>> Step 6: HuggingFace 登入..."
source ~/.zshrc 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$NAS_DIR/venv"
export UV_LINK_MODE=copy
uv run huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential 2>/dev/null || true

# ── Step 7: 下載訓練用 Demo 資料（6 tasks，224×224）─────────────────
log ">>> Step 7: 下載 behavior_224_rgb（6 tasks，含重試機制）..."
uv run python - <<'PY'
import os, time
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

token = os.environ.get("HF_TOKEN")
dest  = "/media/ML_2025/shawn/b1k/data/behavior_224_rgb"
os.makedirs(dest, exist_ok=True)

# task_index 對應（task_name → task-XXXX folder）
# picking_up_trash=1, picking_up_toys=7, tidying_bedroom=18,
# collecting_childrens_toys=21, sorting_household_items=27, clean_up_your_desk=29
task_ids = [1, 7, 18, 21, 27, 29]

allow_patterns = [
    "meta/info.json", "meta/tasks.jsonl",
    "meta/episodes.jsonl", "meta/episodes_stats.jsonl", "README.md",
]
for tid in task_ids:
    folder = f"task-{tid:04d}"
    allow_patterns += [f"data/{folder}/*", f"videos/{folder}/*"]

print(f"下載 task folders: {[f'task-{t:04d}' for t in task_ids]}\n目的地: {dest}", flush=True)

# 重試機制：HF server 端 5xx 錯誤時自動等待後重試
for attempt in range(6):
    try:
        snapshot_download(
            repo_id="IliaLarchenko/behavior_224_rgb",
            repo_type="dataset",
            local_dir=dest,
            allow_patterns=allow_patterns,
            token=token,
        )
        print("訓練資料下載完成")
        break
    except HfHubHTTPError as e:
        code = getattr(e.response, "status_code", 0) if hasattr(e, "response") else 0
        if code in (502, 503, 504) or "Gateway" in str(e) or "timeout" in str(e).lower():
            wait = 2 ** attempt * 15  # 15, 30, 60, 120, 240, 480 秒
            print(f"[Attempt {attempt+1}/6] HF server error ({code}), 等待 {wait}s 後重試...", flush=True)
            if attempt == 5:
                raise
            time.sleep(wait)
        else:
            raise
PY
log "訓練資料下載完成"

# ── Step 8: 下載 Checkpoint 1, 2, 3（含重試）────────────────────────
log ">>> Step 8: 下載 Checkpoints..."
uv run python - <<'PY'
import os, time
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

token = os.environ.get("HF_TOKEN")
nas   = "/media/ML_2025/shawn/b1k/checkpoints"
os.makedirs(nas, exist_ok=True)

def download_with_retry(max_retries=6, **kwargs):
    for attempt in range(max_retries):
        try:
            return snapshot_download(**kwargs)
        except HfHubHTTPError as e:
            code = getattr(e.response, "status_code", 0) if hasattr(e, "response") else 0
            if code in (502, 503, 504) or "Gateway" in str(e) or "timeout" in str(e).lower():
                wait = 2 ** attempt * 15
                print(f"[Attempt {attempt+1}/{max_retries}] server error ({code}), 等 {wait}s...", flush=True)
                if attempt == max_retries - 1: raise
                time.sleep(wait)
            else:
                raise

for ckpt in ["checkpoint_1", "checkpoint_2", "checkpoint_3"]:
    dest = f"{nas}/{ckpt}"
    if os.path.exists(dest) and any(os.scandir(dest)):
        print(f"[SKIP] {ckpt} 已存在"); continue
    print(f"[DOWNLOAD] {ckpt}...", flush=True)
    download_with_retry(
        repo_id="IliaLarchenko/behavior_submission",
        repo_type="model",
        local_dir=dest,
        allow_patterns=[f"{ckpt}/**"],
        token=token,
    )
    print(f"[DONE] {ckpt}")
PY
log "Checkpoints 下載完成"

# ── Step 9: 下載 OmniGibson Assets（評估用，含重試）─────────────────
log ">>> Step 9: 下載 OmniGibson Assets → $NAS_DIR/og_assets..."
OG_DEST="$NAS_DIR/og_assets"
mkdir -p "$OG_DEST"

uv run python - <<PYEOF
import os, time
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError

token = os.environ.get("HF_TOKEN")
base  = "$OG_DEST"

def download_with_retry(max_retries=6, **kwargs):
    for attempt in range(max_retries):
        try:
            return snapshot_download(**kwargs)
        except HfHubHTTPError as e:
            code = getattr(e.response, "status_code", 0) if hasattr(e, "response") else 0
            if code in (401, 403):
                raise  # 權限問題不重試，直接拋出
            if code in (502, 503, 504) or "Gateway" in str(e) or "timeout" in str(e).lower():
                wait = 2 ** attempt * 15
                print(f"[Attempt {attempt+1}/{max_retries}] server error ({code}), 等 {wait}s...", flush=True)
                if attempt == max_retries - 1: raise
                time.sleep(wait)
            else:
                raise

datasets = [
    ("behavior-1k/omnigibson-robot-assets",        "omnigibson-robot-assets"),
    ("behavior-1k/behavior-1k-assets",              "behavior-1k-assets"),
    ("behavior-1k/2025-challenge-task-instances",   "2025-challenge-task-instances"),
]

for repo_id, subdir in datasets:
    dest = os.path.join(base, subdir)
    if os.path.exists(dest) and any(os.scandir(dest)):
        print(f"[SKIP] {subdir} 已存在"); continue
    print(f"[DOWNLOAD] {repo_id}...", flush=True)
    try:
        download_with_retry(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=dest,
            token=token,
        )
        print(f"[DONE] {subdir}")
    except HfHubHTTPError as e:
        code = getattr(e.response, "status_code", 0) if hasattr(e, "response") else 0
        if code in (401, 403):
            print(f"[NEED_ACCESS] {repo_id} 需要申請 HuggingFace 存取權限（HTTP {code}）")
        else:
            print(f"[ERROR] {repo_id}: {e}")
PYEOF
log "OmniGibson Assets 下載完成（或已跳過）"

# ── Step 10: 複製 norm stats assets────────────────────────────────
log ">>> Step 10: 從 Checkpoint 複製 norm stats..."
ASSETS_OUT="$NAS_DIR/outputs/assets/pi_behavior_b1k_fast"
mkdir -p "$ASSETS_OUT"
for CKPT in checkpoint_2 checkpoint_1 checkpoint_3; do
    SRC="$NAS_DIR/checkpoints/$CKPT/assets"
    [ -d "$SRC" ] && cp -rn "$SRC/"* "$ASSETS_OUT/" 2>/dev/null && log "Assets 來自 $CKPT" && break
done

log "=== Phase 1 全部完成 ==="
touch "$NAS_DIR/logs/setup_done.sentinel"
log "Sentinel 建立：Phase 2 fine-tuning 與 baseline eval 將自動啟動"
