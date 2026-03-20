"""
src/data/dataset.py

PyTorch Dataset for the ESC-50 dataset (Environmental Sound Classification).

Repo data layout
----------------
data/
├── audio/                   ← 2 000 .wav files (44.1 kHz, 5 s)
│   ├── 1-100032-A-0.wav
│   └── ...
└── meta/
    └── esc50.csv            ← filename, fold, target, category, ...

Each Dataset item is a dict:
    {
        "waveform":    torch.Tensor  shape (channels, samples),
        "sample_rate": int,
        "label":       torch.Tensor  scalar long  — class index 0–49
        "path":        str           — absolute path to the .wav file
    }

Download
--------
    git clone https://github.com/karoldvl/ESC-50.git data/ESC-50
    # Then point root="data/ESC-50" or symlink data/audio and data/meta.

Fold convention
---------------
ESC-50 defines 5 standard folds.  The canonical evaluation protocol holds
out fold 5 as the test set and trains on folds 1–4.

    train_ds = ESC50Dataset("data", folds=[1, 2, 3, 4])
    test_ds  = ESC50Dataset("data", folds=[5])
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import soundfile as sf
import torch
from torch.utils.data import Dataset

# ── Label vocabulary ──────────────────────────────────────────────────────────
# 50 categories in target-index order (0–49).  Matches the 'target' column of
# esc50.csv exactly; do not reorder.

ESC50_CLASSES: List[str] = [
    # Animals (0–9)
    "dog",
    "rooster",
    "pig",
    "cow",
    "frog",
    "cat",
    "hen",
    "insects",
    "sheep",
    "crow",
    # Natural soundscapes & water (10–19)
    "rain",
    "sea_waves",
    "crackling_fire",
    "crickets",
    "chirping_birds",
    "water_drops",
    "wind",
    "pouring_water",
    "toilet_flush",
    "thunderstorm",
    # Human, non-speech sounds (20–29)
    "crying_baby",
    "sneezing",
    "clapping",
    "breathing",
    "coughing",
    "footsteps",
    "laughing",
    "brushing_teeth",
    "snoring",
    "drinking_sipping",
    # Interior/domestic sounds (30–39)
    "door_wood_knock",
    "mouse_click",
    "keyboard_typing",
    "door_wood_creaks",
    "can_opening",
    "washing_machine",
    "vacuum_cleaner",
    "clock_alarm",
    "clock_tick",
    "glass_breaking",
    # Exterior/urban noises (40–49)
    "helicopter",
    "chainsaw",
    "siren",
    "car_horn",
    "engine",
    "train",
    "church_bells",
    "airplane",
    "fireworks",
    "hand_saw",
]

LABEL2IDX: Dict[str, int] = {lbl: idx for idx, lbl in enumerate(ESC50_CLASSES)}
IDX2LABEL: Dict[int, str] = {idx: lbl for lbl, idx in LABEL2IDX.items()}
NUM_CLASSES: int = len(ESC50_CLASSES)  # 50

# ── Dataset ───────────────────────────────────────────────────────────────────


class ESC50Dataset(Dataset):
    """
    PyTorch Dataset for ESC-50 audio clips.

    Parameters
    ----------
    root : str | Path
        Root directory containing ``audio/`` and ``meta/esc50.csv``.
        Pass the path to your ESC-50 clone, e.g. ``"data"``.
    folds : list[int]
        Which ESC-50 folds to include (1–5).  Use ``[1,2,3,4]`` for training
        and ``[5]`` for the held-out test set.
    transform : callable, optional
        Applied to each raw sample dict before it is returned.
        Signature: ``transform(sample: dict) -> dict``.
        Use ``AudioToAST`` from ``src.data.transforms`` here.
    max_clips : int, optional
        Cap the dataset at *max_clips* examples (handy for smoke-tests).
    """

    def __init__(
        self,
        root: Union[str, Path],
        folds: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        max_clips: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.root = Path(root).expanduser().resolve()
        self.folds = set(folds) if folds is not None else {1, 2, 3, 4, 5}
        self.transform = transform

        # List of (wav_path, label_tensor) tuples
        self._samples: List[Tuple[Path, torch.Tensor]] = []
        self._index()

        if max_clips is not None:
            self._samples = self._samples[:max_clips]

    # ── Indexing ──────────────────────────────────────────────────────────────

    def _index(self) -> None:
        """Parse esc50.csv and collect clips belonging to the requested folds."""
        csv_path = self.root / "meta" / "esc50.csv"
        audio_dir = self.root / "audio"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {csv_path}\n"
                "Expected layout:  <root>/meta/esc50.csv\n"
                "Download ESC-50:  git clone https://github.com/karoldvl/ESC-50.git"
            )
        if not audio_dir.is_dir():
            raise FileNotFoundError(
                f"Audio directory not found: {audio_dir}\n"
                "Expected layout:  <root>/audio/<filename>.wav")

        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                fold = int(row["fold"])
                if fold not in self.folds:
                    continue
                target = int(row["target"])
                filename = row["filename"]
                wav_path = audio_dir / filename
                if not wav_path.exists():
                    import warnings
                    warnings.warn(f"Audio file missing: {wav_path}",
                                  stacklevel=2)
                    continue
                label = torch.tensor(target, dtype=torch.long)
                self._samples.append((wav_path, label))

        if not self._samples:
            raise RuntimeError(
                f"No clips found for folds {sorted(self.folds)} in {csv_path}."
            )

    # ── Dataset protocol ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict:
        wav_path, label = self._samples[idx]

        data, sample_rate = sf.read(str(wav_path), always_2d=True)
        waveform = torch.from_numpy(data.T).float()  # (channels, samples)

        sample = {
            "waveform": waveform,  # (C, T)
            "sample_rate": sample_rate,
            "label": label,  # scalar long — class index 0–49
            "path": str(wav_path),
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    # ── Convenience ───────────────────────────────────────────────────────────

    def class_counts(self) -> Dict[str, int]:
        """Return a dict mapping category name → number of clips."""
        counts: Dict[str, int] = {cls: 0 for cls in ESC50_CLASSES}
        for _, label in self._samples:
            counts[IDX2LABEL[int(label.item())]] += 1
        return counts

    def __repr__(self) -> str:
        return (f"ESC50Dataset(root='{self.root}', "
                f"folds={sorted(self.folds)}, "
                f"n_samples={len(self)}, "
                f"n_classes={NUM_CLASSES})")
