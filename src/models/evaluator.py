"""
src/models/evaluator.py

Load a saved AST model and evaluate it on the ESC-50 test fold (fold 5).

Outputs
-------
- Accuracy (top-1)
- F1 macro / weighted
- Per-class F1 breakdown
- results/<model_stem>_metrics.json

Run from repo root:
    python -m src.models.evaluator
    python -m src.models.evaluator --model-dir models/ast_baseline
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import ESC50Dataset, ESC50_CLASSES
from src.data.transforms import AudioToAST
from src.utils.model_loader import is_onnx_dir, load_model

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_DIR = "models/ast_baseline"
CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
RESULTS_DIR = "results"
BATCH_SIZE = 8
MAX_LENGTH_S = 5.0


def _results_path(model_dir: str, compiled: bool) -> str:
    stem = Path(model_dir).stem
    suffix = "_compiled" if compiled else ""
    return str(Path(RESULTS_DIR) / f"{stem}{suffix}_metrics.json")


def evaluate(model_dir: str, compile_model: bool = False) -> dict:
    device = (torch.device("mps")
              if torch.backends.mps.is_available() else torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    _onnx = is_onnx_dir(model_dir)
    model = load_model(model_dir, device)

    # torch.compile is only applicable to PyTorch nn.Module, not ORT sessions.
    compile_applied = False
    if compile_model:
        if _onnx:
            print(
                "  Note: torch.compile is not applicable to ONNX models — skipping."
            )
        else:
            if device.type == "cuda":
                compile_kwargs = {"mode": "reduce-overhead"}
            else:
                compile_kwargs = {"backend": "aot_eager"}
            print(f"Compiling model with torch.compile({compile_kwargs}) …")
            model = torch.compile(model, **compile_kwargs)
            compile_applied = True

    # ── Load test dataset ─────────────────────────────────────────────────────
    fe_source = model_dir if Path(
        model_dir, "preprocessor_config.json").exists() else CHECKPOINT
    transform = AudioToAST(
        model_checkpoint=fe_source,
        max_length_s=MAX_LENGTH_S,
        padding=True,
    )
    # ESC-50 canonical test set is fold 5.
    test_ds = ESC50Dataset("data", folds=[5], transform=transform)
    print(f"Test samples: {len(test_ds):,}")

    def collate(samples):
        return {
            "input_values": torch.stack([s["input_values"] for s in samples]),
            "labels": torch.stack([s["label"] for s in samples]),
        }

    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, collate_fn=collate)

    # ── Run inference ─────────────────────────────────────────────────────────
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(loader):
            logits = model(batch["input_values"].to(device)).logits
            all_logits.append(logits.cpu().numpy())
            all_labels.append(batch["labels"].numpy())

    logits = np.concatenate(all_logits, axis=0)  # (N, 50)
    labels = np.concatenate(all_labels, axis=0)  # (N,)  class indices

    # ── Metrics ───────────────────────────────────────────────────────────────
    # Softmax → argmax for top-1 prediction
    exp_l = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)  # (N, 50)
    preds = np.argmax(probs, axis=1)  # (N,)
    labels_int = labels.astype(int)

    accuracy = accuracy_score(labels_int, preds)
    f1_macro = f1_score(labels_int, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels_int,
                           preds,
                           average="weighted",
                           zero_division=0)
    per_class_f1 = f1_score(labels_int, preds, average=None, zero_division=0)

    per_class_f1_d = {
        cls: round(float(f1), 4)
        for cls, f1 in zip(ESC50_CLASSES, per_class_f1)
    }

    results = {
        "model_dir": model_dir,
        "compiled": compile_applied,
        "n_test": len(test_ds),
        "accuracy": round(float(accuracy), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "per_class_F1": per_class_f1_d,
    }

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  Accuracy          : {accuracy:.4f}")
    print(f"  F1 (macro)        : {f1_macro:.4f}")
    print(f"  F1 (weighted)     : {f1_weighted:.4f}")
    print(f"{'─'*50}")
    print("  Per-class F1:")
    for cls in ESC50_CLASSES:
        f1 = per_class_f1_d[cls]
        bar = "█" * int(f1 * 20)
        print(f"    {cls:<22}  F1 {f1:.4f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    results_path = _results_path(model_dir, compile_applied)
    os.makedirs(Path(results_path).parent, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to '{results_path}'")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Path to saved model directory",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap model with torch.compile after loading",
    )
    args = parser.parse_args()
    evaluate(args.model_dir, compile_model=args.compile)
