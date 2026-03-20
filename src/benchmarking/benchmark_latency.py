"""
src/benchmarking/benchmark_latency.py

Latency profiler for the AST inference pipeline.

Measures wall-clock time for each stage of a *single-request* (batch=1)
inference, simulating a real serving scenario:

  1. Audio loading    – torchaudio.load()
  2. Preprocessing   – resample + mono downmix + log-mel extraction
                       (AudioToAST / ASTFeatureExtractor)
  3. Model inference – forward pass through the AST model

Additionally runs torch.profiler on the model forward pass for a detailed
operator-level breakdown, and saves a Chrome-trace JSON for visualisation
in chrome://tracing or Perfetto UI.

Statistics reported per stage
------------------------------
  mean, std, p50 (median), p95, min, max  (all in milliseconds)

Usage
-----
    python -m src.benchmarking.benchmark_latency
    python -m src.benchmarking.benchmark_latency --model-dir models/ast_baseline
    python -m src.benchmarking.benchmark_latency --n-requests 200 --no-profiler

CLI flags
---------
  --model-dir    Path to saved model directory     (default: models/ast_baseline)
  --n-requests   Number of timed inferences        (default: 100)
  --n-warmup     Warm-up passes before timing      (default: 10)
  --output       JSON results path                 (default: results/latency_<model>.json)
  --no-profiler  Skip the torch.profiler run
  --compile      Wrap model with torch.compile(mode='reduce-overhead') after loading
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch

from src.data.dataset import ESC50Dataset
from src.data.transforms import AudioToAST
from src.utils.model_loader import is_onnx_dir, load_model

# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_DIR = "models/ast_baseline"
CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
DATA_ROOT = "data"
MAX_LENGTH_S = 5.0
RESULTS_DIR = "results"

# ── Helpers ───────────────────────────────────────────────────────────────────


def _sync(device: torch.device) -> None:
    """Block until all pending ops on the device are finished."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def compute_stats(times_ms: List[float]) -> Dict[str, float]:
    """Return descriptive statistics (in ms) for a list of timings."""
    arr = np.array(times_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "n": int(len(arr)),
    }


def _print_stage(label: str, stats: Dict[str, float]) -> None:
    print(f"  {label:<18}  "
          f"mean {stats['mean_ms']:7.2f} ms  "
          f"p50 {stats['p50_ms']:7.2f} ms  "
          f"p95 {stats['p95_ms']:7.2f} ms  "
          f"std {stats['std_ms']:6.2f} ms")


# ── Main benchmark ────────────────────────────────────────────────────────────


def run_benchmark(
    model_dir: str,
    n_requests: int,
    n_warmup: int,
    output: Optional[str],
    run_profiler: bool,
    compile_model: bool = False,
    device_override: Optional[str] = None,
) -> dict:
    # ── Device ────────────────────────────────────────────────────────────────
    if device_override:
        device = torch.device(device_override)
    else:
        device = (torch.device("mps") if torch.backends.mps.is_available() else
                  torch.device("cuda")
                  if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Device     : {device}")
    print(f"Model dir  : {model_dir}")
    print(f"Requests   : {n_requests}  (+ {n_warmup} warm-up)")

    # ── Model ─────────────────────────────────────────────────────────────────
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

    # ── Feature extractor ─────────────────────────────────────────────────────
    fe_source = (model_dir if Path(
        model_dir, "preprocessor_config.json").exists() else CHECKPOINT)
    transform = AudioToAST(
        model_checkpoint=fe_source,
        max_length_s=MAX_LENGTH_S,
        padding=True,
    )

    # ── Audio file paths (raw dataset, no transform) ──────────────────────────
    # Use fold 5 (the canonical test fold) for benchmarking.
    raw_ds = ESC50Dataset(DATA_ROOT, folds=[5], transform=None)
    samples = raw_ds._samples  # List[Tuple[Path, Tensor]]
    n_clips = len(samples)
    print(f"Test clips : {n_clips:,}\n")

    # ── Warm-up ───────────────────────────────────────────────────────────────
    print(f"Running {n_warmup} warm-up passes …")
    with torch.no_grad():
        for i in range(n_warmup):
            wav_path, label = samples[i % n_clips]
            data, sample_rate = sf.read(str(wav_path), always_2d=True)
            waveform = torch.from_numpy(data.T).float()  # (channels, samples)
            sample = {
                "waveform": waveform,
                "sample_rate": sample_rate,
                "label": label,
                "path": str(wav_path),
            }
            processed = transform(sample)
            input_tensor = processed["input_values"].unsqueeze(0).to(device)
            _sync(device)
            _ = model(input_tensor)
            _sync(device)

    # ── Timed benchmark ───────────────────────────────────────────────────────
    print(f"Timing {n_requests} single-request inferences …\n")

    t_load: List[float] = []
    t_preprocess: List[float] = []
    t_infer: List[float] = []

    with torch.no_grad():
        for i in range(n_requests):
            wav_path, label = samples[i % n_clips]
            path_str = str(wav_path)

            # ── Stage 1 : Audio loading ────────────────────────────────────
            t0 = time.perf_counter()
            data, sample_rate = sf.read(path_str, always_2d=True)
            waveform = torch.from_numpy(data.T).float()  # (channels, samples)
            t1 = time.perf_counter()
            t_load.append((t1 - t0) * 1_000)

            # ── Stage 2 : Preprocessing (resample + mono + mel) ───────────
            raw_sample = {
                "waveform": waveform,
                "sample_rate": sample_rate,
                "label": label,
                "path": path_str,
            }
            t2 = time.perf_counter()
            processed = transform(raw_sample)
            t3 = time.perf_counter()
            t_preprocess.append((t3 - t2) * 1_000)

            # ── Stage 3 : Model inference (batch size = 1) ─────────────────
            input_tensor = processed["input_values"].unsqueeze(0).to(device)
            _sync(device)
            t4 = time.perf_counter()
            _ = model(input_tensor)
            _sync(device)
            t5 = time.perf_counter()
            t_infer.append((t5 - t4) * 1_000)

            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1:>4}/{n_requests}]  "
                      f"load {t_load[-1]:6.1f} ms  "
                      f"preprocess {t_preprocess[-1]:6.1f} ms  "
                      f"infer {t_infer[-1]:6.1f} ms")

    # ── Aggregate results ─────────────────────────────────────────────────────
    totals = [a + b + c for a, b, c in zip(t_load, t_preprocess, t_infer)]

    results: dict = {
        "model_dir": model_dir,
        "compiled": compile_applied,
        "device": str(device),
        "n_requests": n_requests,
        "n_warmup": n_warmup,
        "batch_size": 1,
        "audio_loading": compute_stats(t_load),
        "preprocessing": compute_stats(t_preprocess),
        "model_inference": compute_stats(t_infer),
        "total_pipeline": compute_stats(totals),
    }

    # ── Pretty-print ──────────────────────────────────────────────────────────
    sep = "─" * 72
    print(f"\n{sep}")
    print(
        f"  {'Stage':<18}  {'mean':>10}   {'p50':>10}   {'p95':>10}   {'std':>8}"
    )
    print(sep)
    for label, key in [
        ("Audio loading", "audio_loading"),
        ("Preprocessing", "preprocessing"),
        ("Model inference", "model_inference"),
        ("Total pipeline", "total_pipeline"),
    ]:
        _print_stage(label, results[key])
    print(sep)

    # ── Torch Profiler ────────────────────────────────────────────────────────
    if run_profiler:
        print("\nRunning torch.profiler on a single forward pass …")

        # Prepare one input (reuse already-warmed-up transform)
        wav_path, label = samples[0]
        data, sample_rate = sf.read(str(wav_path), always_2d=True)
        waveform = torch.from_numpy(data.T).float()  # (channels, samples)
        raw_sample = {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "label": label,
            "path": str(wav_path),
        }
        input_tensor = (
            transform(raw_sample)["input_values"].unsqueeze(0).to(device))

        # Profiler activities
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        # Note: MPS operations are traced through CPU activity on Apple Silicon.

        with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_flops=True,
                with_modules=True,
        ) as prof:
            with torch.no_grad():
                _ = model(input_tensor)

        # Top-20 operators by CPU time
        table = prof.key_averages(group_by_input_shape=True).table(
            sort_by="cpu_time_total",
            row_limit=20,
        )
        print(
            "\n[torch.profiler] Top-20 ops by CPU time (single forward pass)\n"
        )
        print(table)

        # Chrome trace for Perfetto / chrome://tracing
        model_stem = Path(model_dir).stem
        compile_tag = "_compiled" if compile_applied else ""
        trace_path = os.path.join(
            RESULTS_DIR, f"profiler_trace_{model_stem}{compile_tag}.json")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        prof.export_chrome_trace(trace_path)
        print(f"\nChrome trace saved → '{trace_path}'")
        print("  Open with: https://ui.perfetto.dev  or  chrome://tracing\n")

        results["profiler_trace"] = trace_path

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if output is None:
        model_stem = Path(model_dir).stem
        compile_tag = "_compiled" if compile_applied else ""
        output = os.path.join(RESULTS_DIR,
                              f"latency_{model_stem}{compile_tag}.json")

    os.makedirs(Path(output).parent, exist_ok=True)
    with open(output, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved → '{output}'")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Benchmark per-stage inference latency for the AST pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Path to saved model directory",
    )
    parser.add_argument(
        "--n-requests",
        type=int,
        default=100,
        help="Number of timed single-request inferences",
    )
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="Number of warm-up passes before timing begins",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: results/latency_<model>.json)",
    )
    parser.add_argument(
        "--no-profiler",
        action="store_true",
        help="Skip the torch.profiler detailed operator breakdown",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help=
        "Wrap model with torch.compile(mode='reduce-overhead') after loading",
    )
    parser.add_argument(
        "--device",
        default=None,
        help=("Force a specific device (e.g. 'cpu', 'cuda', 'mps'). "
              "Defaults to auto-detection: mps > cuda > cpu."),
    )
    args = parser.parse_args()

    run_benchmark(
        model_dir=args.model_dir,
        n_requests=args.n_requests,
        n_warmup=args.n_warmup,
        output=args.output,
        run_profiler=not args.no_profiler,
        compile_model=args.compile,
        device_override=args.device,
    )
