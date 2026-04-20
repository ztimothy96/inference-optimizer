#!/usr/bin/env bash
# upload_benchmark.sh
#
# Upload repository source and benchmark assets to a Runpod SSH instance,
# then optionally run setup_remote.sh on the pod.
#
# Usage:
#   ./upload_benchmark.sh --model-dir models/ast_baseline \
#       --host root@<ip-address> --key ~/.ssh/<key-name> --port <port>
#
# What is uploaded
# ----------------
#   src/                          → project source
#   pyproject.toml                → build config
#   setup_remote.sh               → one-time environment setup
#   data/meta/esc50.csv           → label metadata
#   data/audio/5-*.wav            → all fold-5 validation audio clips
#   <model-dir>/                  → always written to models/benchmark_model/
#                                   (checkpoint-* subdirs are excluded)
#
# Required flags
# --------------
#   --model-dir   Local path to the model directory to benchmark
#   --host        SSH destination, i.e. root@<ip-address>
#   --key         SSH private key path
#   --port        SSH port
#
# Optional flags
# --------------
#   --remote-dir  Remote working dir      (default: /root/inference-optimizer)
#   --setup       Run setup_remote.sh on the pod after uploading

set -euo pipefail

# ── Initial values ────────────────────────────────────────────────────────────

MODEL_DIR=""
HOST=""
SSH_KEY=""
PORT=""
REMOTE_DIR="/root/inference-optimizer"
RUN_SETUP=false

# ── Helpers ───────────────────────────────────────────────────────────────────

info()  { echo "[upload] $*"; }
ok()    { echo "[upload] OK: $*"; }
die()   { echo "[upload] ERROR: $*" >&2; exit 1; }

usage() {
    grep '^#' "$0" | sed 's/^# \{0,1\}//'
    exit 0
}

# ── Argument parsing ──────────────────────────────────────────────────────────

[[ $# -eq 0 ]] && usage

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-dir)  MODEL_DIR="$2";  shift 2 ;;
        --host)       HOST="$2";       shift 2 ;;
        --key)        SSH_KEY="$2";    shift 2 ;;
        --port)       PORT="$2";       shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --setup)      RUN_SETUP=true;  shift   ;;
        -h|--help)    usage ;;
        *) die "Unknown flag: $1" ;;
    esac
done

# ── Validation ────────────────────────────────────────────────────────────────

[[ -z "$MODEL_DIR" ]] && die "--model-dir is required"
[[ -z "$HOST" ]]      && die "--host is required"
[[ -z "$SSH_KEY" ]]   && die "--key is required"
[[ -z "$PORT" ]]      && die "--port is required"
[[ -d "$MODEL_DIR" ]] || die "Model directory not found: $MODEL_DIR"
[[ -f "$SSH_KEY" ]]   || die "SSH key not found: $SSH_KEY"
[[ -d "src" ]]        || die "Run this script from the project root (src/ not found)"

AUDIO_DIR="data/audio"
META_CSV="data/meta/esc50.csv"
[[ -d "$AUDIO_DIR" ]]  || die "Audio directory not found: $AUDIO_DIR"
[[ -f "$META_CSV" ]]   || die "Metadata CSV not found: $META_CSV"

# ── SSH helpers ───────────────────────────────────────────────────────────────
# Transfer method: tar | ssh "tar xf - -C dest"

SSH_OPTS=(-i "$SSH_KEY" -p "$PORT"
          -o StrictHostKeyChecking=no
          -o BatchMode=yes
          -T)           # -T: no PTY; the gateway banner appears on stdout but
                        #     does not affect the stdin pipe we use for uploads.

remote_exec() {
    ssh "${SSH_OPTS[@]}" "$HOST" "$*"
}

# tar_up <remote-extract-dir> <local-path>…
#   Creates a gzip'd tar archive of the listed local paths and extracts it on
#   the remote under <remote-extract-dir> (relative to REMOTE_DIR).
tar_up() {
    local dest="$1"; shift
    tar czf - "$@" \
        | ssh "${SSH_OPTS[@]}" "$HOST" \
              "tar xzf - -C '${REMOTE_DIR}/${dest}'"
}

# ── 0. Ensure remote directory exists ─────────────────────────────────────────

info "==> [0/4] Preparing remote directory …"
remote_exec "mkdir -p '${REMOTE_DIR}/src' \
                       '${REMOTE_DIR}/data/meta' \
                       '${REMOTE_DIR}/data/audio' \
                       '${REMOTE_DIR}/models/benchmark_model'"
ok "Remote directory layout ready"

# ── 1. Source tree ────────────────────────────────────────────────────────────

info "==> [1/4] Uploading source tree …"
# Paths in the archive are src/…, pyproject.toml, setup_remote.sh; they land
# correctly when extracted relative to REMOTE_DIR.
tar_up "." src/ pyproject.toml setup_remote.sh
ok "Source tree uploaded"

# ── 2. Data assets ────────────────────────────────────────────────────────────

info "==> [2/4] Uploading data assets …"

# Metadata CSV (archive path is data/meta/esc50.csv → extracted under REMOTE_DIR)
tar_up "." "$META_CSV"
ok "Metadata CSV uploaded"

# Select all fold-5 audio files (ESC-50 filenames start with "{fold}-").
# Use a while-read loop instead of mapfile for bash 3.2 compatibility (macOS).
AUDIO_FILES=()
while IFS= read -r f; do
    AUDIO_FILES+=("$f")
done < <(find "$AUDIO_DIR" -maxdepth 1 -type f -name "5-*.wav" | sort)

if [[ ${#AUDIO_FILES[@]} -eq 0 ]]; then
    die "No fold-5 .wav files found in $AUDIO_DIR (expected filenames like 5-*.wav)"
fi

info "Uploading ${#AUDIO_FILES[@]} audio clip(s) …"
# Archive paths are data/audio/…; extracted under REMOTE_DIR.
tar_up "." "${AUDIO_FILES[@]}"
ok "${#AUDIO_FILES[@]} audio clip(s) uploaded"

# ── 3. Model (always written to models/benchmark_model/) ─────────────────────

info "==> [3/4] Uploading model → models/benchmark_model/ …"
# Wipe and recreate the destination so switching models never leaves stale
# weights behind (mirrors the semantics of rsync --delete).
remote_exec "rm -rf '${REMOTE_DIR}/models/benchmark_model' && \
             mkdir -p '${REMOTE_DIR}/models/benchmark_model'"

# Archive the model contents with -C so archive paths are ./config.json etc.,
# extracted directly into benchmark_model/ with no extra nesting.
# No compression (tar cf, not czf): model weights are dense binary blobs that
# compress negligibly, so gzip only wastes CPU time on both ends.
# Exclude checkpoint-* subdirectories: they contain duplicate weights plus
# optimizer states (~433 MB each) that are irrelevant for inference.

# Compute bytes to be transferred (mirrors the tar --exclude above) so pv can
# show a meaningful percentage and ETA.  stat -f%z is macOS; falls back to 0
# (no ETA) on systems where it is unavailable.
UPLOAD_BYTES=$(find "$MODEL_DIR" -not -path "*/checkpoint-*" -type f \
               -exec stat -f%z {} \; 2>/dev/null | awk '{s+=$1} END {print s+0}')

if command -v pv &>/dev/null; then
    tar cf - -C "$MODEL_DIR" --exclude='./checkpoint-*' . \
        | pv -s "$UPLOAD_BYTES" -N "  model" \
        | ssh "${SSH_OPTS[@]}" "$HOST" \
              "tar xf - -C '${REMOTE_DIR}/models/benchmark_model'"
else
    info "  (tip: brew install pv to get a live progress bar)"
    tar cf - -C "$MODEL_DIR" --exclude='./checkpoint-*' . \
        | ssh "${SSH_OPTS[@]}" "$HOST" \
              "tar xf - -C '${REMOTE_DIR}/models/benchmark_model'"
fi
ok "Model uploaded ($(du -sh "$MODEL_DIR" | cut -f1) total local size, checkpoints excluded)"

# ── 4. Optional: run setup on the pod ────────────────────────────────────────

if $RUN_SETUP; then
    info "==> [4/4] Running setup_remote.sh on pod …"
    remote_exec "cd '${REMOTE_DIR}' && bash setup_remote.sh"
    ok "Remote setup complete"
else
    info "==> [4/4] Skipping remote setup (pass --setup to run setup_remote.sh)"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Upload complete. To benchmark on the pod:"
echo ""
echo "  ssh -i $SSH_KEY -p $PORT $HOST"
echo "  cd $REMOTE_DIR"
echo "  source .venv/bin/activate"
echo "  python -m src.benchmarking.benchmark_latency \\"
echo "      --model-dir models/benchmark_model --device cuda"
echo "================================================================"
