#!/usr/bin/env bash
# upload_benchmark.sh
#
# Uploads the repository source and the assets needed to run
# benchmark_latency.py on a remote ARM instance.
#
# What gets uploaded
# ------------------
#   src/               Python package
#   pyproject.toml     Package metadata and dependencies (pip install -e .)
#   setup_remote.sh    One-time environment setup script for the remote
#   data/meta/esc50.csv                  Dataset index
#   data/audio/<fold-5 files only>       400 clips, ~168 MiB
#   models/benchmark_model/              The requested model (no checkpoints)
#
# The model is always written to the same fixed path on the remote
# (models/benchmark_model/) so successive runs overwrite it, keeping
# disk/memory usage low.
#
# Usage
# -----
#   ./upload_benchmark.sh --model-dir models/ast_baseline --host ec2-user@1.2.3.4
#   ./upload_benchmark.sh --model-dir models/ast_baseline_onnx \
#       --host ec2-user@1.2.3.4 --key ~/.ssh/my-key.pem --port 22
#
# After uploading (first time only — sets up Python 3.12 venv + deps):
#   ssh ec2-user@<IP>
#   cd ~/inference-optimizer
#   bash setup_remote.sh
#
# To benchmark (every run):
#   source .venv/bin/activate
#   python -m src.benchmarking.benchmark_latency \
#       --model-dir models/benchmark_model --device cpu --no-profiler
#
# Requirements (local machine)
# ----------------------------
#   rsync, ssh, python3

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
REMOTE_BASE="~/inference-optimizer"
REMOTE_MODEL_DIR="models/benchmark_model"
SSH_PORT=22
SSH_KEY=""
DATA_ROOT="data"
ESC50_FOLD=5

# ── Argument parsing ──────────────────────────────────────────────────────────
usage() {
    echo "Usage: $0 --model-dir <path> --host <user@host> [--key <pem>] [--port <n>] [--remote-dir <path>]"
    echo ""
    echo "  --model-dir    Local path to the model directory to upload (required)"
    echo "  --host         SSH destination, e.g. ubuntu@1.2.3.4             (required)"
    echo "  --key          Path to SSH private key, e.g. ~/.ssh/my-key.pem  (optional)"
    echo "  --port         SSH port                                          (default: 22)"
    echo "  --remote-dir   Base directory on the remote host                 (default: ~/inference-optimizer)"
    exit 1
}

MODEL_DIR=""
SSH_HOST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)  MODEL_DIR="$2";    shift 2 ;;
        --host)       SSH_HOST="$2";     shift 2 ;;
        --key)        SSH_KEY="$2";      shift 2 ;;
        --port)       SSH_PORT="$2";     shift 2 ;;
        --remote-dir) REMOTE_BASE="$2";  shift 2 ;;
        -h|--help)    usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$MODEL_DIR" ]] && { echo "Error: --model-dir is required."; usage; }
[[ -z "$SSH_HOST"  ]] && { echo "Error: --host is required.";      usage; }
[[ ! -d "$MODEL_DIR" ]] && { echo "Error: model directory '$MODEL_DIR' not found."; exit 1; }

# ── SSH / rsync helpers ───────────────────────────────────────────────────────
SSH_OPTS="-p $SSH_PORT -o StrictHostKeyChecking=no"
[[ -n "$SSH_KEY" ]] && SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
RSYNC_SSH="ssh $SSH_OPTS"

rsync_upload() {
    # rsync_upload <extra rsync flags…> <local> <remote-path>
    rsync -avz --progress -e "$RSYNC_SSH" "$@"
}

remote_run() {
    # remote_run <command>  — run a command on the remote host
    ssh $SSH_OPTS "$SSH_HOST" "$@"
}

# ── 1. Create remote directory layout ─────────────────────────────────────────
echo ""
echo "==> [1/4] Creating remote directory structure …"
remote_run "mkdir -p $REMOTE_BASE/models $REMOTE_BASE/data/audio $REMOTE_BASE/data/meta"

# ── 2. Upload source code and config files ────────────────────────────────────
echo ""
echo "==> [2/4] Uploading source code …"
rsync_upload \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.venv' \
    --exclude='*.egg-info' \
    --filter=':- .gitignore' \
    src \
    pyproject.toml \
    setup_remote.sh \
    "$SSH_HOST:$REMOTE_BASE/"

# ── 3. Upload the model (no checkpoints) ──────────────────────────────────────
echo ""
echo "==> [3/4] Uploading model '$(basename "$MODEL_DIR")' → remote '$REMOTE_MODEL_DIR' …"
echo "    (Checkpoint subdirectories are excluded to save disk space)"

# Delete the remote model dir first so stale files from a previous model don't linger.
remote_run "rm -rf $REMOTE_BASE/$REMOTE_MODEL_DIR && mkdir -p $REMOTE_BASE/$REMOTE_MODEL_DIR"

rsync_upload \
    --exclude='checkpoint-*' \
    "$MODEL_DIR/" \
    "$SSH_HOST:$REMOTE_BASE/$REMOTE_MODEL_DIR/"

# ── 4. Upload ESC-50 data (fold 5 only) ───────────────────────────────────────
echo ""
echo "==> [4/4] Uploading ESC-50 fold $ESC50_FOLD audio clips and metadata …"

CSV_PATH="$DATA_ROOT/meta/esc50.csv"
[[ ! -f "$CSV_PATH" ]] && { echo "Error: '$CSV_PATH' not found. Is \$DATA_ROOT set correctly?"; exit 1; }

# Build a temp file-list of fold-5 filenames for rsync --files-from.
TMPFILE="$(mktemp /tmp/esc50_fold5_XXXXXX.txt)"
trap 'rm -f "$TMPFILE"' EXIT

python3 - <<'PYEOF' > "$TMPFILE"
import csv, sys, os

data_root = os.environ.get("DATA_ROOT", "data")
fold      = int(os.environ.get("ESC50_FOLD", "5"))
csv_path  = os.path.join(data_root, "meta", "esc50.csv")

with open(csv_path, newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if int(row["fold"]) == fold:
            print(row["filename"])
PYEOF

N_CLIPS="$(wc -l < "$TMPFILE" | tr -d ' ')"
echo "    Found $N_CLIPS clips in fold $ESC50_FOLD."

# Upload metadata CSV
rsync_upload \
    "$DATA_ROOT/meta/esc50.csv" \
    "$SSH_HOST:$REMOTE_BASE/data/meta/esc50.csv"

# Upload only fold-5 audio files (--files-from paths are relative to $DATA_ROOT/audio/)
rsync_upload \
    --files-from="$TMPFILE" \
    "$DATA_ROOT/audio/" \
    "$SSH_HOST:$REMOTE_BASE/data/audio/"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "====================================================================="
echo "Upload complete."
echo ""
echo "On the remote, run:"
echo "  ssh $SSH_OPTS $SSH_HOST"
echo "  cd $REMOTE_BASE"
echo "  bash setup_remote.sh          # first time only"
echo "  source .venv/bin/activate"
echo "  python -m src.benchmarking.benchmark_latency \\"
echo "      --model-dir $REMOTE_MODEL_DIR --device cpu --no-profiler"
echo "====================================================================="
