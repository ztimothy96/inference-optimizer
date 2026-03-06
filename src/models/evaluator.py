"""
src/models/evaluator.py

Load a saved AST model and evaluate it on the IRMAS test set.

Outputs
-------
- mAP (macro)          — ranking-based, multi-label fair
- F1 macro / micro     — threshold-based (default 0.5)
- Hamming loss         — fraction of wrong label slots
- AUC-ROC macro        — area under ROC, macro-averaged
- Per-class AP & AUC   — instrument-level breakdown
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
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
)
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForAudioClassification
from tqdm import tqdm

from src.data.dataset import IRMASDataset, IRMAS_CLASSES, NUM_CLASSES
from src.data.transforms import IRMAStoAST

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_DIR = "models/ast_baseline"
CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
RESULTS_PATH = "results/baseline_metrics.json"
BATCH_SIZE = 8
MAX_LENGTH_S = 3.0
THRESHOLD = 0.5  # sigmoid probability threshold for positive prediction


def evaluate(model_dir: str) -> dict:
    device = (torch.device("mps")
              if torch.backends.mps.is_available() else torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))
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

    logits = np.concatenate(all_logits, axis=0)  # (N, 11)
    labels = np.concatenate(all_labels, axis=0)  # (N, 11)  multi-hot

    # ── Metrics ───────────────────────────────────────────────────────────────
    # Sigmoid: independent per-class probabilities (BCEWithLogitsLoss model)
    probs = 1.0 / (1.0 + np.exp(-logits))  # (N, NUM_CLASSES)
    preds = (probs >= THRESHOLD).astype(int)  # binary predictions
    labels_int = labels.astype(int)

    # mAP — ranking-based, fair for single- and multi-label samples
    mAP = average_precision_score(labels_int, probs, average="macro")

    # Per-class AP
    per_class_ap = average_precision_score(labels_int, probs, average=None)
    per_class_ap_d = {
        cls: round(float(ap), 4)
        for cls, ap in zip(IRMAS_CLASSES, per_class_ap)
    }

    # F1 — threshold-based
    f1_macro = f1_score(labels_int, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels_int, preds, average="micro", zero_division=0)

    # Hamming loss — fraction of individual label slots predicted incorrectly
    h_loss = hamming_loss(labels_int, preds)

    # Per-class AUC-ROC
    try:
        per_class_auc = roc_auc_score(labels_int, probs, average=None)
        auc_macro = float(np.mean(per_class_auc))
    except ValueError:
        per_class_auc = [float("nan")] * NUM_CLASSES
        auc_macro = float("nan")
    per_class_auc_d = {
        cls: round(float(auc), 4)
        for cls, auc in zip(IRMAS_CLASSES, per_class_auc)
    }

    results = {
        "model_dir": model_dir,
        "n_test": len(test_ds),
        "mAP": round(float(mAP), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_micro": round(float(f1_micro), 4),
        "hamming": round(float(h_loss), 4),
        "auc_macro": round(auc_macro, 4),
        "per_class_AP": per_class_ap_d,
        "per_class_AUC": per_class_auc_d,
    }

    # ── Print ─────────────────────────────────────────────────────────────────
    print(f"\n{'─'*40}")
    print(f"  mAP   (macro)     : {mAP:.4f}")
    print(f"  F1    (macro)     : {f1_macro:.4f}")
    print(f"  F1    (micro)     : {f1_micro:.4f}")
    print(f"  Hamming loss      : {h_loss:.4f}")
    print(f"  AUC   (macro)     : {auc_macro:.4f}")
    print(f"{'─'*40}")
    print("  Per-class AP  /  AUC:")
    for cls in IRMAS_CLASSES:
        ap = per_class_ap_d[cls]
        auc = per_class_auc_d[cls]
        bar = "█" * int(ap * 20)
        print(f"    {cls:>3}  AP {ap:.4f}  AUC {auc:.4f}  {bar}")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(Path(RESULTS_PATH).parent, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to '{RESULTS_PATH}'")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Path to saved model directory (default: models/ast_baseline)",
    )
    args = parser.parse_args()
    evaluate(args.model_dir)
