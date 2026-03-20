"""
src/models/export_onnx.py

Export a fine-tuned AST model to ONNX format.

The exported graph accepts a single input:
    input_values : float32  shape (batch, time_frames, mel_bins)

and returns:
    logits       : float32  shape (batch, num_classes)

Usage
-----
    python -m src.models.export_onnx
    python -m src.models.export_onnx --model-dir models/ast_baseline \\
                                     --output-dir models/ast_onnx

CLI flags
---------
  --model-dir   Path to saved HF model directory  (default: models/ast_baseline)
  --output-dir  Directory for model.onnx output   (default: models/ast_onnx)
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
from transformers import ASTFeatureExtractor, AutoModelForAudioClassification

from src.data.transforms import TARGET_SR

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_DIR = "models/ast_baseline"
DEFAULT_OUTPUT_DIR = "models/ast_onnx"
DEFAULT_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"

MAX_LENGTH_S = 5.0  # must match the value used during training / evaluation
OPSET_VERSION = 17  # ONNX opset 17 is the minimum recommended for Transformers

# ── Wrapper ───────────────────────────────────────────────────────────────────


class _ASTWrapper(torch.nn.Module):
    """
    Thin wrapper around the HF ASTForAudioClassification model.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        return self.model(input_values=input_values).logits


# ── Export function ───────────────────────────────────────────────────────────


def export_to_onnx(model_dir: str, output_dir: str) -> str:
    """
    Export a saved HF AST model to ONNX.

    Parameters
    ----------
    model_dir  : Path to the saved HF model directory.
    output_dir : Destination directory.  Will be created if it does not exist.
                 Writes ``model.onnx`` and copies ``preprocessor_config.json``
                 so the directory is self-contained.

    Returns
    -------
    str
        Absolute path to the exported ``model.onnx`` file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    onnx_path = str(output_path / "model.onnx")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from '{model_dir}' …")
    model = AutoModelForAudioClassification.from_pretrained(model_dir)
    model.eval()
    wrapped = _ASTWrapper(model)

    # ── Build dummy input from the real feature extractor ────────────────────
    # Run a silent clip through ASTFeatureExtractor so the dummy tensor has
    # the exact shape the model expects — no hard-coding required.
    fe_source = (model_dir if
                 (Path(model_dir) /
                  "preprocessor_config.json").exists() else DEFAULT_CHECKPOINT)
    feature_extractor = ASTFeatureExtractor.from_pretrained(fe_source)
    max_samples = int(MAX_LENGTH_S * TARGET_SR)
    dummy_audio = np.zeros(max_samples, dtype=np.float32)

    encoded = feature_extractor(
        dummy_audio,
        sampling_rate=TARGET_SR,
        max_length=max_samples,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    dummy_input = encoded["input_values"]  # (1, time_frames, mel_bins)
    print(f"Input shape : {tuple(dummy_input.shape)}  "
          f"(batch=1, time_frames, mel_bins)")

    # ── Export ────────────────────────────────────────────────────────────────
    print(f"Exporting ONNX (opset {OPSET_VERSION}) → '{onnx_path}' …")
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            (dummy_input, ),
            onnx_path,
            input_names=["input_values"],
            output_names=["logits"],
            dynamic_axes={
                "input_values": {
                    0: "batch_size"
                },
                "logits": {
                    0: "batch_size"
                },
            },
            opset_version=OPSET_VERSION,
        )

    # Copy feature-extractor config alongside the model so the output
    # directory is self-contained (same convention as HF model dirs).
    fe_cfg = Path(model_dir) / "preprocessor_config.json"
    if fe_cfg.exists():
        shutil.copy(fe_cfg, output_path / "preprocessor_config.json")

    print(f"ONNX model saved → '{onnx_path}'")
    return onnx_path


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export AST model to ONNX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Path to saved HF model directory",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write model.onnx",
    )
    args = parser.parse_args()
    export_to_onnx(args.model_dir, args.output_dir)
