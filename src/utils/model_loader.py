"""
src/utils/model_loader.py

Unified model loader that returns either a PyTorch HuggingFace model or a
thin OnnxModelWrapper, both with an identical call interface:

    model   = load_model(model_dir, device)
    outputs = model(input_values_tensor)   # any device
    logits  = outputs.logits               # torch.Tensor on CPU

This lets benchmark_latency.py and evaluator.py stay device/format-agnostic
— the same inference loop works for PyTorch, quanto-quantised, and ONNX
(fp32 or int8) model directories.

ONNX backend selection
----------------------
ONNX models always run through ORT (``onnxruntime``):

- **CUDA**  → ``CUDAExecutionProvider`` (if available), fallback to CPU.
- **MPS / CPU** → ``CPUExecutionProvider``.

ORT has no MPS execution provider.  On Apple Silicon the ONNX track is
primarily a **server / CPU deployment track**; for local speed on MPS use
the native PyTorch model instead (it runs on Apple's GPU via MPS directly).

Notes on CoreML EP
------------------
ORT's CoreML Execution Provider crashes with a ``!model_path.empty()``
assertion when it partitions the graph into in-memory sub-graph protos
(unfixed as of ORT 1.24.3 on macOS arm64).

``coremltools`` dropped ONNX as a conversion source in v7 (supported
sources are now ``pytorch``, ``tensorflow``, and ``milinternal`` only), so
converting ONNX → CoreML as a workaround is not available either.

Both limitations are bypassed by running ONNX on CPU via ORT, which is
stable and well-tested.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Union

import torch
from transformers import AutoModelForAudioClassification

# ── Helpers ───────────────────────────────────────────────────────────────────


def is_onnx_dir(model_dir: str) -> bool:
    """Return True if *model_dir* contains a ``model.onnx`` file."""
    return (Path(model_dir) / "model.onnx").exists()


# ── ONNX wrapper ──────────────────────────────────────────────────────────────


class OnnxModelWrapper:
    """
    Wraps an ``onnxruntime.InferenceSession`` with a HuggingFace-compatible
    call interface.

    Usage
    -----
    >>> wrapper = OnnxModelWrapper(session)
    >>> outputs = wrapper(input_values_tensor)   # torch.Tensor, any device
    >>> logits  = outputs.logits                 # torch.Tensor on CPU

    The wrapper is intentionally minimal:
    - ``__call__``   runs the ORT session and returns a namespace with ``.logits``
    - ``eval()``     no-op, returns self
    - ``to(device)`` no-op, returns self  (ORT manages its own device)
    """

    def __init__(self, session) -> None:
        self._session = session

    def __call__(self, input_values: torch.Tensor) -> SimpleNamespace:
        # ORT expects a numpy array on CPU regardless of the input device.
        arr = input_values.detach().cpu().numpy()
        logits_np = self._session.run(["logits"], {"input_values": arr})[0]
        return SimpleNamespace(logits=torch.from_numpy(logits_np))

    def eval(self) -> "OnnxModelWrapper":
        return self

    def to(self, device) -> "OnnxModelWrapper":
        return self  # ORT manages its own device via ExecutionProviders


# ── Unified loader ────────────────────────────────────────────────────────────


def load_model(
    model_dir: str,
    device: torch.device,
) -> Union[torch.nn.Module, OnnxModelWrapper]:
    """
    Load a model from *model_dir*, auto-detecting PyTorch vs ONNX format.

    - If ``model_dir/model.onnx`` exists → returns an ``OnnxModelWrapper``
      backed by an ORT InferenceSession.
    - Otherwise → loads with ``AutoModelForAudioClassification``, calls
      ``eval()`` and ``to(device)``, and returns the nn.Module.

    Parameters
    ----------
    model_dir : str
        Model directory — either a HuggingFace saved model or a directory
        produced by ``export_to_onnx()`` / ``quantize_onnx_int8()``.
    device : torch.device
        Target device for PyTorch models.  For ONNX models the device
        determines the ORT ExecutionProvider:
          - ``cuda`` → CUDAExecutionProvider (if available), then CPU fallback
          - ``mps`` / ``cpu`` → CPUExecutionProvider
            (ORT has no MPS provider; see module docstring for details)

    Returns
    -------
    nn.Module | OnnxModelWrapper
        Ready-to-use model.  ``eval()`` and ``to(device)`` are already called.
    """
    if is_onnx_dir(model_dir):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required to run ONNX models.\n"
                "Install with:  pip install onnxruntime") from exc

        onnx_path = str(Path(model_dir).resolve() / "model.onnx")

        # Choose execution providers based on available hardware.
        # MPS falls through to CPU: ORT has no MPS EP, and CoreML EP has an
        # unfixed crash on macOS arm64 (see module docstring).
        available = ort.get_available_providers()
        if device.type == "cuda" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        print(f"Loading ONNX model from '{onnx_path}' …")
        print(f"  ORT providers : {providers}")
        session = ort.InferenceSession(onnx_path, providers=providers)
        return OnnxModelWrapper(session)

    else:
        print(f"Loading PyTorch model from '{model_dir}' …")
        model = AutoModelForAudioClassification.from_pretrained(model_dir)
        model.eval()
        model.to(device)
        return model
