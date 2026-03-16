"""
src/models/train.py

Fine-tune the AST model on the ESC-50 dataset.

Steps
-----
1. Load ESC-50 train (folds 1–4) and test (fold 5) splits via
   ESC50Dataset + AudioToAST transform.
2. Load the pre-trained AST model, replacing its classification head
   for 50 single-label classes.
3. Train with the HuggingFace Trainer (CrossEntropyLoss via
   problem_type="single_label_classification").
4. Evaluate on the held-out test fold.
5. Save the model to OUTPUT_DIR.

Run from the repo root after `pip install -e .`:
    python -m src.models.train
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoModelForAudioClassification,
    TrainingArguments,
    Trainer,
)

from src.data.dataset import ESC50Dataset, ESC50_CLASSES, NUM_CLASSES
from src.data.transforms import AudioToAST

# ── Config ────────────────────────────────────────────────────────────────────

CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
OUTPUT_DIR = "./models/ast_esc50"
BATCH_SIZE = 4  # small batch to fit in MPS / 16 GB memory
GRAD_ACCUM_STEPS = 4  # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 16
EPOCHS = 5
LR = 1e-5
SEED = 42
MAX_LENGTH_S = 5.0  # ESC-50 clips are exactly 5 seconds

# Unfreeze only the last N transformer layers + classifier to save memory.
# The AST encoder has 12 layers total; 4 is a good accuracy/memory tradeoff.
N_UNFREEZE_LAYERS = 4

# ── 1. Transform and Datasets ─────────────────────────────────────────────────

transform = AudioToAST(
    model_checkpoint=CHECKPOINT,
    max_length_s=MAX_LENGTH_S,
    padding=True,
)

# ESC-50 canonical split: folds 1–4 for training, fold 5 for test.
train_ds = ESC50Dataset("data", folds=[1, 2, 3, 4], transform=transform)
test_ds = ESC50Dataset("data", folds=[5], transform=transform)

print(f"Train: {len(train_ds):,} | Test: {len(test_ds):,}")

# ── 2. Model ──────────────────────────────────────────────────────────────────

id2label = {i: lbl for i, lbl in enumerate(ESC50_CLASSES)}
label2id = {lbl: i for i, lbl in id2label.items()}

model = AutoModelForAudioClassification.from_pretrained(
    CHECKPOINT,
    num_labels=NUM_CLASSES,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,  # replaces the AudioSet head (527 classes)
    problem_type="single_label_classification",  # → CrossEntropyLoss
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


def collate_fn(samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Stack input_values and labels into a batch."""
    return {
        "input_values": torch.stack([s["input_values"] for s in samples]),
        "labels": torch.stack([s["label"] for s in samples]),
    }


# ── 4. Metrics ────────────────────────────────────────────────────────────────


def compute_metrics(eval_pred):
    """
    Single-label metrics for environmental sound classification.

    eval_pred.predictions : (N, NUM_CLASSES)  raw logits
    eval_pred.label_ids   : (N,)              integer class indices
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    labels_int = labels.astype(int)

    accuracy = accuracy_score(labels_int, preds)
    f1_macro = f1_score(labels_int, preds, average="macro", zero_division=0)
    f1_weighted = f1_score(labels_int,
                           preds,
                           average="weighted",
                           zero_division=0)

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }


# ── 5. Train ──────────────────────────────────────────────────────────────────

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LR,
    warmup_ratio=0.06,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=50,
    fp16=False,  # set True if your hardware supports it
    seed=SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    data_collator=collate_fn,
    compute_metrics=compute_metrics,
)

print("Starting fine-tuning …")
trainer.train()

# ── 6. Evaluate, Save ──────────────────────────────────────────────────────────

results = trainer.evaluate(test_ds)
print(f"\nTest results: {results}")

trainer.save_model(OUTPUT_DIR)
print(f"\nModel saved to '{OUTPUT_DIR}/'")
