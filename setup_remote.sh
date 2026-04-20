#!/usr/bin/env bash
# setup_remote.sh
#
# One-time setup script for a Runpod GPU pod (Ubuntu / apt).
# Run from the project root after uploading with upload_benchmark.sh:
#
#   bash setup_remote.sh
#
# What this does
# --------------
#   1. Installs python3.12 via apt (deadsnakes PPA) if not already present
#   2. Creates a Python 3.12 virtual environment at .venv/
#   3. Upgrades pip to >= 24.0 inside the venv
#   4. Installs the project and all dependencies via pip install -e .
#   5. On CUDA hosts: installs onnxruntime-gpu and tensorrt-cu12, and
#      appends LD_LIBRARY_PATH to the venv activate script so the TensorRT
#      shared libraries are found automatically.

set -euo pipefail

PYTHON_BIN="python3.12"
VENV_DIR=".venv"
MIN_PIP="24.0"

# ── Helpers ───────────────────────────────────────────────────────────────────

info()    { echo "[setup] $*" >&2; }
success() { echo "[setup] OK: $*" >&2; }

maybe_sudo() {
    if [[ $EUID -eq 0 ]]; then
        "$@"
    else
        sudo "$@"
    fi
}

require_root_or_sudo() {
    if [[ $EUID -ne 0 ]] && ! sudo -n true 2>/dev/null; then
        echo "[setup] ERROR: This script needs sudo to install system packages."
        echo "        Run as root or ensure your user has passwordless sudo."
        exit 1
    fi
}

apt_install_if_missing() {
    local cmd="$1"   # command to check with 'command -v'
    local pkg="$2"   # apt package name to install if missing
    if command -v "$cmd" &>/dev/null; then
        success "$cmd already installed ($(command -v "$cmd"))"
    else
        info "Installing $pkg via apt …"
        maybe_sudo apt-get install -y "$pkg"
        success "$pkg installed"
    fi
}

# ── 1. System packages ────────────────────────────────────────────────────────
info "==> [1/3] Checking system packages …"
require_root_or_sudo

if command -v python3.12 &>/dev/null; then
    success "python3.12 already installed ($(command -v python3.12))"
else
    info "Adding deadsnakes PPA and installing python3.12 …"
    # add-apt-repository requires apt_pkg which is broken in many container
    # images, so we add the PPA manually instead.
    maybe_sudo apt-get install -y curl gnupg lsb-release ca-certificates
    curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xF23C5A6CF475977595C89F51BA6932366A755776" \
        | maybe_sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/deadsnakes.gpg
    echo "deb https://ppa.launchpadcontent.net/deadsnakes/ppa/ubuntu $(lsb_release -cs) main" \
        | maybe_sudo tee /etc/apt/sources.list.d/deadsnakes.list > /dev/null
    maybe_sudo apt-get update -y
    maybe_sudo apt-get install -y python3.12 python3.12-venv python3.12-dev
    success "python3.12 installed"
fi

# ── 2. Virtual environment ────────────────────────────────────────────────────
info "==> [2/3] Setting up virtual environment …"

if [[ -d "$VENV_DIR" ]]; then
    success "Virtual environment already exists at $VENV_DIR/ — skipping creation"
else
    info "Creating $VENV_DIR/ with $PYTHON_BIN …"
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
    success "Virtual environment created at $VENV_DIR/"
fi

# Activate for the rest of this script.
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# ── 3. pip upgrade ────────────────────────────────────────────────────────────
CURRENT_PIP=$(pip --version | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)
info "pip version in venv: $CURRENT_PIP (minimum required: $MIN_PIP)"

# Compare major version numbers only; upgrade if current major < minimum major.
CURRENT_MAJOR=$(echo "$CURRENT_PIP" | cut -d. -f1)
MIN_MAJOR=$(echo "$MIN_PIP" | cut -d. -f1)
if [[ "$CURRENT_MAJOR" -ge "$MIN_MAJOR" ]]; then
    success "pip $CURRENT_PIP >= $MIN_PIP — no upgrade needed"
else
    info "Upgrading pip to >= $MIN_PIP …"
    pip install --upgrade "pip>=$MIN_PIP"
    success "pip upgraded to $(pip --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+'| head -1)"
fi

# ── 4. Project install ────────────────────────────────────────────────────────
info "==> [3/3] Installing project dependencies …"

# PyPI only carries CPU torch wheels; CUDA-variant wheels must come from the
# PyTorch index. We detect the max CUDA version the driver supports via
# nvidia-smi and pre-install torch/torchaudio from the right index URL so that
# the subsequent `pip install -e .` doesn't pull a mismatched CUDA build.
get_torch_index_url() {
    if ! command -v nvidia-smi &>/dev/null; then
        info "nvidia-smi not found — will install CPU-only PyTorch"
        echo "https://download.pytorch.org/whl/cpu"
        return
    fi
    # nvidia-smi header line: "CUDA Version: 12.8"
    local cuda_ver
    cuda_ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1)
    if [[ -z "$cuda_ver" ]]; then
        info "Could not parse CUDA version from nvidia-smi — defaulting to cu128"
        echo "https://download.pytorch.org/whl/cu128"
        return
    fi
    local major minor
    major=$(echo "$cuda_ver" | cut -d. -f1)
    minor=$(echo "$cuda_ver" | cut -d. -f2)
    info "Detected driver CUDA support: $cuda_ver → PyTorch index tag: cu${major}${minor}"
    echo "https://download.pytorch.org/whl/cu${major}${minor}"
}

if python -c "import torch" &>/dev/null; then
    success "torch already installed ($(python -c 'import torch; print(torch.__version__)'))"
else
    TORCH_INDEX_URL=$(get_torch_index_url)
    info "Pre-installing torch and torchaudio from $TORCH_INDEX_URL …"
    pip install torch torchaudio --index-url "$TORCH_INDEX_URL"
fi

# --config-settings editable_mode=compat forces setuptools to use a .pth file
# that adds the project root to sys.path. The default "strict" editable mode
# uses import hooks that can fail to expose top-level packages like 'src'.
pip install -e . --config-settings editable_mode=compat

# onnxruntime (CPU-only) is listed in pyproject.toml for portability, but on
# a CUDA host we need onnxruntime-gpu to get CUDAExecutionProvider.
# Installing onnxruntime-gpu replaces the CPU package automatically.
if command -v nvidia-smi &>/dev/null; then
    info "CUDA host detected — replacing onnxruntime with onnxruntime-gpu …"
    pip install --upgrade onnxruntime-gpu
    success "onnxruntime-gpu installed"

    # TensorRT — required for the ORT TensorRT Execution Provider.
    # tensorrt-cu12 ships libnvinfer.so.10 and friends via pip; no system-level
    # TensorRT installation is needed.  We skip if already present to avoid a
    # ~1.5 GB re-download on re-runs.
    if python -c "import tensorrt" &>/dev/null; then
        success "tensorrt already installed ($(python -c 'import tensorrt; print(tensorrt.__version__)'))"
    else
        info "Installing tensorrt-cu12 (~1.5 GB, this may take a few minutes) …"
        pip install tensorrt-cu12
        success "tensorrt-cu12 installed"
    fi

    # Append the TensorRT library path to the venv's activate script so that
    # `source .venv/bin/activate` always exposes libnvinfer.so.10 to ORT.
    # We write a guard comment so the block is only appended once.
    ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
    if ! grep -q "# [tensorrt LD_LIBRARY_PATH]" "$ACTIVATE_SCRIPT"; then
        info "Patching $ACTIVATE_SCRIPT with TensorRT LD_LIBRARY_PATH …"
        cat >> "$ACTIVATE_SCRIPT" << 'EOF'

# [tensorrt LD_LIBRARY_PATH] — added by setup_remote.sh
_trt_lib_dir=$(python -c "import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))" 2>/dev/null || true)
if [[ -n "$_trt_lib_dir" ]]; then
    export LD_LIBRARY_PATH="$_trt_lib_dir:${LD_LIBRARY_PATH:-}"
fi
unset _trt_lib_dir
EOF
        success "LD_LIBRARY_PATH patch written to $ACTIVATE_SCRIPT"
    else
        success "$ACTIVATE_SCRIPT already patched — skipping"
    fi
fi

success "Project installed"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Setup complete. To run the benchmark:"
echo ""
echo "  source $VENV_DIR/bin/activate"
echo "  python -m src.benchmarking.benchmark_latency \\"
echo "      --model-dir models/benchmark_model --device cuda"
echo "================================================================"
