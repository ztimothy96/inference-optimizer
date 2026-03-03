"""
src/models/evaluator.py

Load a saved AST model and evaluate it on the IRMAS test set.

Outputs
-------
- Accuracy (argmax, single-label style)
- Mean Average Precision (mAP, multi-label fair)
- Per-class AP breakdown
- results/baseline_metrics.json

Run from repo root:
    python -m src.models.evaluator
    python -m src.models.evaluator --model-dir models/ast_baseline
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import softmax
from sklearn.metrics import accuracy_score, average_precision_score
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForAudioClassification
from tqdm import tqdm

from src.data.dataset import IRMASDataset, IRMAS_CLASSES, NUM_CLASSES
from src.data.transforms import IRMAStoAST

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_DIR = "models/ast_baseline"
CHECKPOINT        = "MIT/ast-finetuned-audioset-10-10-0.4593"
RESULTS_PATH      = "results/baseline_metrics.json"
BATCH_SIZE        = 8
MAX_LENGTH_S      = 3.0


def evaluate(model_dir: str) -> dict:
    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    print(f"Device: {device}")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model from '{model_dir}' …")
    model = AutoModelForAudioClassification.from_pretrained(model_dir)
    model.eval()
    model.to(device)

    # ── Load test dataset ─────────────────────────────────────────────────────
    transform = IRMAStoAST(
        model_checkpoint=CHECKPOINT,
        max_length_s=MAX_LENGTH_S,
        padding=True,
    )
    test_ds = IRMASDataset("data/test", split="test", transform=transform)
    print(f"Test samples: {len(test_ds):,}")

    def collate(samples):
        return {
            "input_values": torch.stack([s["input_values"] for s in samples]),
            "labels":       torch.stack([s["label"]        for s in samples]),
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

    logits = np.concatenate(all_logits, axis=0)   # (N, 11)
    labels = np.concatenate(all_labels, axis=0)   # (N, 11)  multi-hot

    # ── Metrics ───────────────────────────────────────────────────────────────
    probs = softmax(logits, axis=-1)              # (N, 11)

    # Accuracy — argmax comparison (interpretable, but biased on multi-label)
    preds = np.argmax(logits, axis=-1)
    refs  = np.argmax(labels, axis=-1)
    accuracy = accuracy_score(refs, preds)

    # mAP — ranking-based, fair for both single- and multi-label samples
    mAP = average_precision_score(labels, probs, average="macro")

    # Per-class AP (useful for spotting which instruments are hardest)
    per_class_ap = average_precision_score(labels, probs, average=None)
    per_class = {cls: round(float(ap), 4)
                 for cls, ap in zip(IRMAS_CLASSES, per_class_ap)}

    results = {
        "model_dir":   model_dir,
        "n_test":      len(test_ds),
        "accuracy":    round(float(accuracy), 4),
        "mAP":         round(float(mAP), 4),
        "per_class_AP": per_class,
    }

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"  Accuracy (argmax) : {accuracy:.4f}")
    print(f"  mAP (macro)       : {mAP:.4f}")
    print(f"{'─'*40}")
    print("  Per-class AP:")
    for cls, ap in sorted(per_class.items(), key=lambda x: -x[1]):
        bar = "█" * int(ap * 20)
        print(f"    {cls:>3}  {ap:.4f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(Path(RESULTS_PATH).parent, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to '{RESULTS_PATH}'")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", default=DEFAULT_MODEL_DIR,
        help="Path to saved model directory (default: models/ast_baseline)",
    )
    args = parser.parse_args()
    evaluate(args.model_dir)
