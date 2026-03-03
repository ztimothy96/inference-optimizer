"""
src/data/dataset.py

PyTorch Dataset for the IRMAS dataset (Instrument Recognition in Musical
Audio Signals).

Repo data layout
----------------
data/
├── train/                   ← split="train" root
│   ├── cel/                 ← one sub-directory per instrument class
│   │   ├── [cel][cla]0001__1.wav
│   │   └── ...
│   ├── cla/
│   └── ...  (flu, gac, gel, org, pia, sax, tru, vio, voi)
│
└── test/                    ← split="test" root  (flat directory)
    ├── <song_name>-<n>.wav
    ├── <song_name>-<n>.txt  ← one instrument label per line (tab-stripped)
    └── ...

Each Dataset item is a dict:
    {
        "waveform":    torch.Tensor  shape (channels, samples),
        "sample_rate": int,
        "label":       torch.Tensor  shape (NUM_CLASSES,)  — multi-hot
        "path":        str           — absolute path to the .wav file
    }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torchaudio
from torch.utils.data import Dataset

# ── Label vocabulary ──────────────────────────────────────────────────────────

IRMAS_CLASSES: List[str] = [
    "cel",  # cello
    "cla",  # clarinet
    "flu",  # flute
    "gac",  # acoustic guitar
    "gel",  # electric guitar
    "org",  # organ
    "pia",  # piano
    "sax",  # saxophone
    "tru",  # trumpet
    "vio",  # violin
    "voi",  # voice (singing)
]

LABEL2IDX: Dict[str, int] = {lbl: idx for idx, lbl in enumerate(IRMAS_CLASSES)}
IDX2LABEL: Dict[int, str] = {idx: lbl for lbl, idx in LABEL2IDX.items()}
NUM_CLASSES: int = len(IRMAS_CLASSES)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _labels_to_multihot(labels: Sequence[str]) -> torch.Tensor:
    """Convert a list of instrument label strings to a multi-hot tensor."""
    vec = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    for lbl in labels:
        lbl = lbl.strip().lower()
        if lbl in LABEL2IDX:
            vec[LABEL2IDX[lbl]] = 1.0
        else:
            raise ValueError(
                f"Unknown IRMAS label '{lbl}'. Expected one of {IRMAS_CLASSES}."
            )
    return vec


def _parse_train_path(wav_path: Path) -> str:
    """
    Infer the instrument label from the parent directory name.

    Training clips live in   <root>/<instrument>/<filename>.wav
    so the immediate parent directory IS the label.
    """
    return wav_path.parent.name.lower()


def _parse_test_annotation(txt_path: Path) -> List[str]:
    """
    Read labels from a test-set annotation .txt file.

    Each non-empty line contains one instrument abbreviation, e.g.:
        cel
        vio
    """
    labels: List[str] = []
    with open(txt_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                labels.append(line)
    if not labels:
        raise ValueError(f"No labels found in annotation file: {txt_path}")
    return labels


# ── Dataset ───────────────────────────────────────────────────────────────────


class IRMASDataset(Dataset):
    """
    PyTorch Dataset for IRMAS audio clips.

    Parameters
    ----------
    root : str | Path
        Root directory.
        - For split="train" pass ``data/train/`` (contains one sub-dir per
          instrument class).
        - For split="test"  pass ``data/test/``  (flat directory of .wav /
          .txt pairs).
    split : {"train", "test"}
        Which portion of the dataset to load.
    transform : callable, optional
        A callable applied to each raw sample dict before it is returned.
        Signature: ``transform(sample: dict) -> dict``.
        Use ``IRMAStoAST`` from ``src.data.transforms`` here.
    max_clips : int, optional
        If set, only the first *max_clips* examples are kept (useful for
        quick smoke-tests).
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        max_clips: Optional[int] = None,
    ) -> None:
        super().__init__()
        if split not in ("train", "test"):
            raise ValueError(
                f"split must be 'train' or 'test', got '{split}'.")

        self.root = Path(root).expanduser().resolve()
        self.split = split
        self.transform = transform

        # List of (wav_path, label_tensor) tuples
        self._samples: List[Tuple[Path, torch.Tensor]] = []

        if split == "train":
            self._index_train()
        else:
            self._index_test()

        if max_clips is not None:
            self._samples = self._samples[:max_clips]

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _index_train(self) -> None:
        """
        Walk each per-instrument sub-directory and collect .wav files.
        Expected layout: data/train/<instrument>/*.wav
        """
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Training root not found: {self.root}\n"
                "Expected layout: data/train/<instrument>/*.wav")

        found_classes: List[str] = []
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name.lower()
            if class_name not in LABEL2IDX:
                # Skip non-instrument directories (e.g. __MACOSX)
                continue
            found_classes.append(class_name)
            label_tensor = _labels_to_multihot([class_name])
            for wav_file in sorted(class_dir.glob("*.wav")):
                self._samples.append((wav_file, label_tensor))

        if not self._samples:
            raise RuntimeError(
                f"No .wav files found under {self.root}. "
                f"Expected sub-directories named after IRMAS classes: {IRMAS_CLASSES}"
            )

    def _index_test(self) -> None:
        """
        Collect (wav, txt) pairs from the flat data/test/ directory.
        Every .wav file must have a sibling .txt annotation file whose
        lines each contain one instrument abbreviation (tab-stripped).
        Expected layout: data/test/<song_name>-<n>.{wav,txt}
        """
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Test root not found: {self.root}\n"
                "Expected layout: data/test/<song_name>-<n>.wav (flat directory)"
            )

        wav_files = sorted(self.root.glob("*.wav"))
        if not wav_files:
            raise RuntimeError(f"No .wav files found under {self.root}.")

        missing_annotations: List[str] = []
        for wav_path in wav_files:
            txt_path = wav_path.with_suffix(".txt")
            if not txt_path.exists():
                missing_annotations.append(str(wav_path))
                continue
            labels = _parse_test_annotation(txt_path)
            label_tensor = _labels_to_multihot(labels)
            self._samples.append((wav_path, label_tensor))

        if missing_annotations:
            import warnings
            warnings.warn(
                f"{len(missing_annotations)} .wav file(s) skipped because no "
                f"matching .txt annotation was found. First few:\n" +
                "\n".join(missing_annotations[:5]),
                stacklevel=2,
            )

        if not self._samples:
            raise RuntimeError(
                "No valid (wav + txt) pairs found. "
                "Check that annotation .txt files exist alongside the .wav files."
            )

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict:
        wav_path, label = self._samples[idx]

        waveform, sample_rate = torchaudio.load(str(wav_path))

        sample = {
            "waveform": waveform,  # (C, T)
            "sample_rate": sample_rate,
            "label": label,  # (NUM_CLASSES,) multi-hot
            "path": str(wav_path),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ── Convenience ───────────────────────────────────────────────────────────

    def class_counts(self) -> Dict[str, int]:
        """Return a dict mapping instrument name → number of clips."""
        counts: Dict[str, int] = {cls: 0 for cls in IRMAS_CLASSES}
        for _, label in self._samples:
            for idx, val in enumerate(label):
                if val == 1.0:
                    counts[IDX2LABEL[idx]] += 1
        return counts

    def __repr__(self) -> str:
        return (f"IRMASDataset(split='{self.split}', "
                f"root='{self.root}', "
                f"n_samples={len(self)}, "
                f"n_classes={NUM_CLASSES})")
