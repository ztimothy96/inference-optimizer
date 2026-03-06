"""
src/models/train.py

Fine-tune the AST model on the IRMAS dataset.

Steps
-----
1. Load IRMAS train / test splits via IRMASDataset + IRMAStoAST transform.
2. Carve a validation split from the training data.
3. Load the pre-trained AST model, replacing its classification head.
4. Train with the HuggingFace Trainer.
5. Evaluate on the held-out test set.
6. Save the model to OUTPUT_DIR.

Run from the repo root after `pip install -e .`:
    python -m src.models.train
"""

import numpy as np
import torch
from torch.utils.data import random_split
from dataclasses import dataclass
from typing import Any, Dict, List
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    roc_auc_score,
)

from transformers import (
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)

from src.data.dataset import IRMASDataset, IRMAS_CLASSES, NUM_CLASSES
from src.data.transforms import IRMAStoAST

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
OUTPUT_DIR = "./models/ast_baseline"
BATCH_SIZE = 4  # small batch to fit in MPS / 16 GB memory
GRAD_ACCUM_STEPS = 4  # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 16
EPOCHS = 5
LR = 1e-5
VAL_FRACTION = 0.1  # share of training clips used for validation
SEED = 42
MAX_LENGTH_S = 3.0  # IRMAS training clips are exactly 3 seconds

# Unfreeze only the last N transformer layers + classifier to save memory.
# The AST encoder has 12 layers total; 4 is a good accuracy/memory tradeoff.
N_UNFREEZE_LAYERS = 4

# ── 1. Transform and Datasets ─────────────────────────────────────────────────

transform = IRMAStoAST(
    model_checkpoint=CHECKPOINT,
    max_length_s=MAX_LENGTH_S,
    padding=True,
)

full_train = IRMASDataset("data/train", split="train", transform=transform)
test_ds = IRMASDataset("data/test", split="test", transform=transform)

n_val = int(len(full_train) * VAL_FRACTION)
n_train = len(full_train) - n_val
train_ds, val_ds = random_split(
    full_train,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED),
)

print(
    f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}"
)

# ── 2. Model ──────────────────────────────────────────────────────────────────

id2label = {i: lbl for i, lbl in enumerate(IRMAS_CLASSES)}
label2id = {lbl: i for i, lbl in id2label.items()}

model = AutoModelForAudioClassification.from_pretrained(
    CHECKPOINT,
    num_labels=NUM_CLASSES,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,  # replaces the AudioSet head (527 classes)
    problem_type="multi_label_classification",  # → BCEWithLogitsLoss
)

# Freeze all parameters first, then selectively unfreeze.
for param in model.parameters():
    param.requires_grad = False

# Unfreeze: the last N_UNFREEZE_LAYERS transformer encoder layers,
#           the final layer norm, and the classifier head.
n_encoder_layers = len(model.audio_spectrogram_transformer.encoder.layer)
first_unfrozen = n_encoder_layers - N_UNFREEZE_LAYERS

for i, layer in enumerate(model.audio_spectrogram_transformer.encoder.layer):
    if i >= first_unfrozen:
        for param in layer.parameters():
            param.requires_grad = True

for param in model.audio_spectrogram_transformer.layernorm.parameters():
    param.requires_grad = True

for param in model.classifier.parameters():
    param.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} "
      f"({100 * trainable / total:.1f}%)  "
      f"[encoder layers {first_unfrozen}–{n_encoder_layers - 1} + head]")

# ── 3. Data collator ──────────────────────────────────────────────────────────


@dataclass
class IRMASCollator:
    """
    Collate a list of IRMASDataset samples into batched tensors.

    Each sample dict has:
        "input_values": torch.Tensor  (time_frames, mel_bins)
        "label":        torch.Tensor  (NUM_CLASSES,)  multi-hot float32
    """

    def __call__(self, samples: List[Dict[str,
                                          Any]]) -> Dict[str, torch.Tensor]:
        input_values = torch.stack([s["input_values"] for s in samples])
        labels = torch.stack([s["label"] for s in samples])
        return {"input_values": input_values, "labels": labels}


# ── 4. Metrics ────────────────────────────────────────────────────────────────

THRESHOLD = 0.5  # sigmoid probability threshold for positive label prediction


def compute_metrics(eval_pred):
    """
    Multi-label metrics for instrument recognition.

    eval_pred.predictions : (N, NUM_CLASSES)  raw logits
    eval_pred.label_ids   : (N, NUM_CLASSES)  multi-hot float32
    """
    logits, labels = eval_pred
    labels_int = labels.astype(int)  # int for sklearn multi-label functions

    probs = 1.0 / (1.0 + np.exp(-logits))  # (N, NUM_CLASSES)
    preds = (probs >= THRESHOLD).astype(int)  # (N, NUM_CLASSES)

    mAP = average_precision_score(labels_int, probs, average="macro")
    f1_macro = f1_score(labels_int, preds, average="macro", zero_division=0)
    f1_micro = f1_score(labels_int, preds, average="micro", zero_division=0)
    h_loss = hamming_loss(labels_int, preds)
    try:
        per_class_auc = roc_auc_score(labels_int, probs, average=None)
        auc_macro = float(np.mean(per_class_auc))
    except ValueError:
        per_class_auc = [float("nan")] * NUM_CLASSES
        auc_macro = float("nan")

    metrics = {
        "mAP": float(mAP),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "hamming": float(h_loss),
        "auc_macro": auc_macro,
    }
    # Per-class AUC entries, e.g. "auc_cel", "auc_cla", …
    for cls, auc in zip(IRMAS_CLASSES, per_class_auc):
        metrics[f"auc_{cls}"] = float(auc)

    return metrics


# ── 5. Train ──────────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,  # effective batch = 4 * 4 = 16
    learning_rate=LR,
    warmup_ratio=0.06,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="mAP",
    logging_steps=50,
    fp16=False,  # set True if your hardware supports it
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    data_collator=IRMASCollator(),
    compute_metrics=compute_metrics,
)

print("Starting fine-tuning …")
trainer.train()

# ── 6. Evaluate, Save ──────────────────────────────────────────────────────────

results = trainer.evaluate(test_ds)
print(f"\nTest results: {results}")

trainer.save_model(OUTPUT_DIR)
print(f"\nModel saved to '{OUTPUT_DIR}/'")
