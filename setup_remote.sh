#!/usr/bin/env bash
# setup_remote.sh
#
# One-time setup script for the EC2 instance (Amazon Linux 2023 / dnf).
# Run from the project root after uploading with upload_benchmark.sh:
#
#   bash setup_remote.sh
#
# What this does
# --------------
#   1. Installs python3.12 via dnf if not already present
#   2. Creates a Python 3.12 virtual environment at .venv/
#   3. Upgrades pip to >= 24.0 inside the venv
#   4. Installs the project and all dependencies via pip install -e .

set -euo pipefail

PYTHON_BIN="python3.12"
VENV_DIR=".venv"
MIN_PIP="24.0"

# ── Helpers ───────────────────────────────────────────────────────────────────

info()    { echo "[setup] $*"; }
success() { echo "[setup] OK: $*"; }

require_root_or_sudo() {
    if [[ $EUID -ne 0 ]] && ! sudo -n true 2>/dev/null; then
        echo "[setup] ERROR: This script needs sudo to install system packages."
        echo "        Run as root or ensure your user has passwordless sudo."
        exit 1
    fi
}

dnf_install_if_missing() {
    local cmd="$1"   # command to check with 'command -v'
    local pkg="$2"   # dnf package name to install if missing
    if command -v "$cmd" &>/dev/null; then
        success "$cmd already installed ($(command -v "$cmd"))"
    else
        info "Installing $pkg via dnf …"
        sudo dnf install "$pkg" -y
        success "$pkg installed"
    fi
}

# ── 1. System packages ────────────────────────────────────────────────────────
info "==> [1/3] Checking system packages …"
require_root_or_sudo

dnf_install_if_missing python3.12 python3.12

# ── 2. Virtual environment ────────────────────────────────────────────────────
info "==> [2/3] Setting up virtual environment …"

if [[ -d "$VENV_DIR" ]]; then
    success "Virtual environment already exists at $VENV_DIR/ — skipping creation"
else
    info "Creating $VENV_DIR/ with $PYTHON_BIN …"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "Virtual environment created at $VENV_DIR/"
fi

# Activate for the rest of this script.
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# ── 3. pip upgrade ────────────────────────────────────────────────────────────
CURRENT_PIP=$(pip --version | grep -oE '[0-9]+\.[0-9]+' | head -1)
info "pip version in venv: $CURRENT_PIP (minimum required: $MIN_PIP)"

# Simple major.minor comparison: upgrade if current < minimum.
if python3 -c "
from importlib.metadata import version
from packaging.version import Version
import sys
try:
    cur = Version('$CURRENT_PIP')
    req = Version('$MIN_PIP')
    sys.exit(0 if cur >= req else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    success "pip $CURRENT_PIP >= $MIN_PIP — no upgrade needed"
else
    info "Upgrading pip to >= $MIN_PIP …"
    pip install --upgrade "pip>=$MIN_PIP"
    success "pip upgraded to $(pip --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+'| head -1)"
fi

# ── 4. Project install ────────────────────────────────────────────────────────
info "==> [3/3] Installing project dependencies …"
# --config-settings editable_mode=compat forces setuptools to use a .pth file
# that adds the project root to sys.path. The default "strict" editable mode
# uses import hooks that can fail to expose top-level packages like 'src'.
# soundfile is declared in pyproject.toml and installed automatically here;
# it serves as the torchaudio audio I/O backend since torchcodec has no
# pre-built wheel for linux/aarch64.
pip install -e . --config-settings editable_mode=compat
success "Project installed"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Setup complete. To run the benchmark:"
echo ""
echo "  source $VENV_DIR/bin/activate"
echo "  python -m src.benchmarking.benchmark_latency \\"
echo "      --model-dir models/benchmark_model --device cpu --no-profiler"
echo "================================================================"
