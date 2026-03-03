"""
src/data/transforms.py

Transform that converts a raw IRMAS sample dict (as returned by IRMASDataset)
into the tensor format expected by the AST (Audio Spectrogram Transformer)
model from Hugging Face.

Typical usage
-------------
>>> from src.data.dataset import IRMASDataset
>>> from src.data.transforms import IRMAStoAST
>>>
>>> transform = IRMAStoAST()
>>> ds = IRMASDataset("data/train", split="train", transform=transform)
>>> sample = ds[0]
>>> sample["input_values"].shape   # (1024, 128) — log-mel spectrogram patch
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torchaudio
from transformers import ASTFeatureExtractor

# AST was pre-trained (and fine-tuned) at 16 kHz mono.
TARGET_SR: int = 16_000

# Default checkpoint used throughout this project.
DEFAULT_CHECKPOINT: str = "MIT/ast-finetuned-audioset-10-10-0.4593"


class IRMAStoAST:
    """
    Callable transform: raw waveform → AST ``input_values`` tensor.

    Applies the following pipeline to each sample:
        1. **Resample** to 16 kHz (if needed).
        2. **Downmix** to mono (mean across channels).
        3. **Feature extraction** via ``ASTFeatureExtractor`` — produces a
           log-mel spectrogram of shape ``(time_frames, mel_bins)``,
           e.g. ``(1024, 128)`` for a 10-second clip.

    The transform is designed to be passed to ``IRMASDataset(transform=...)``.
    It receives and returns a sample ``dict``, augmenting it with the
    ``"input_values"`` key while keeping ``"label"`` and ``"path"`` intact.

    Parameters
    ----------
    model_checkpoint : str
        Hugging Face model ID or local path used to load ``ASTFeatureExtractor``.
    max_length_s : float
        Maximum clip duration in **seconds** used when padding/truncating.
        IRMAS training clips are exactly 3 s; the feature extractor's internal
        default is 10 s (AudioSet).  Set to 3.0 for tighter, faster training.
    padding : bool
        Whether to zero-pad clips shorter than ``max_length_s``.
        Keep ``True`` so all tensors in a batch share the same shape.
    return_numpy : bool
        If ``True`` the ``"input_values"`` value is a plain ``numpy.ndarray``
        instead of a ``torch.Tensor``.  Useful when the HF ``Trainer`` is
        used, which expects numpy arrays for data collation.
    """

    def __init__(
        self,
        model_checkpoint: str = DEFAULT_CHECKPOINT,
        max_length_s: float = 3.0,
        padding: bool = True,
        return_numpy: bool = False,
    ) -> None:
        self.feature_extractor: ASTFeatureExtractor = (
            ASTFeatureExtractor.from_pretrained(model_checkpoint))
        self.max_length_s = max_length_s
        self.padding = padding
        self.return_numpy = return_numpy

        # Pre-build a resampler cache to avoid recreating transforms repeatedly.
        self._resamplers: Dict[int, torchaudio.transforms.Resample] = {}

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_resampler(self, orig_sr: int) -> torchaudio.transforms.Resample:
        if orig_sr not in self._resamplers:
            self._resamplers[orig_sr] = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=TARGET_SR)
        return self._resamplers[orig_sr]

    def _preprocess_waveform(self, waveform: torch.Tensor,
                             sample_rate: int) -> torch.Tensor:
        """
        Return a 1-D float32 numpy array at TARGET_SR, ready for the
        feature extractor.

        Parameters
        ----------
        waveform : torch.Tensor  shape (C, T)
        sample_rate : int

        Returns
        -------
        torch.Tensor  shape (T',) at 16 kHz
        """
        # 1. Resample
        if sample_rate != TARGET_SR:
            resampler = self._get_resampler(sample_rate)
            waveform = resampler(waveform)

        # 2. Downmix to mono: (C, T) → (T,)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        # 3. Cast to float32 (torchaudio may return float32 already, but be safe)
        return waveform.to(torch.float32)

    # ── Main entry point ──────────────────────────────────────────────────────

    def __call__(self, sample: Dict) -> Dict:
        """
        Transform a raw sample dict from ``IRMASDataset``.

        Parameters
        ----------
        sample : dict
            Must contain ``"waveform"`` (torch.Tensor, shape C×T) and
            ``"sample_rate"`` (int).  All other keys are passed through.

        Returns
        -------
        dict
            Same keys as the input, plus ``"input_values"`` containing
            the log-mel spectrogram as a ``torch.Tensor`` of shape
            ``(time_frames, mel_bins)``, e.g. ``(128, 128)`` for 3-second
            clips at 16 kHz.
        """
        waveform: torch.Tensor = sample["waveform"]
        sample_rate: int = sample["sample_rate"]

        mono_waveform = self._preprocess_waveform(waveform, sample_rate)

        # ASTFeatureExtractor expects a plain Python list or numpy array.
        audio_np = mono_waveform.numpy()

        max_length_samples = int(self.max_length_s * TARGET_SR)

        encoded = self.feature_extractor(
            audio_np,
            sampling_rate=TARGET_SR,
            max_length=max_length_samples,
            truncation=True,
            padding="max_length" if self.padding else False,
            return_tensors="pt" if not self.return_numpy else "np",
        )

        # encoded["input_values"] shape: (1, time_frames, mel_bins)
        # Squeeze the batch dimension so each sample is (time_frames, mel_bins).
        input_values = encoded["input_values"]
        if not self.return_numpy:
            input_values = input_values.squeeze(0)  # (T_frames, mel_bins)
        else:
            input_values = input_values[0]  # same, numpy

        out = {k: v for k, v in sample.items() if k not in ("waveform", )}
        out["input_values"] = input_values
        return out

    def __repr__(self) -> str:
        return (f"IRMAStoAST("
                f"checkpoint='{self.feature_extractor.name_or_path}', "
                f"max_length_s={self.max_length_s}, "
                f"target_sr={TARGET_SR})")
