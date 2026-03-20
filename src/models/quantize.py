"""
src/models/quantize.py

Two independent INT8 quantization tracks for the AST model.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Track quanto (PyTorch-native, MPS / CUDA / CPU)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Applies weight-only INT8 quantization directly to the PyTorch model.
  The activations remain in float32; only the stored weight matrices are
  compressed to INT8, which halves the model's memory footprint and
  accelerates linear layers on supporting hardware.

  Key steps:
    1. quantize(model, weights=qint8)   – replace weight tensors with
                                          INT8 QuantizedLinear modules
    2. freeze(model)                    – commit the quantization: fold the
                                          scale/zero-point into the weights
                                          and make the model serialisable
    3. model.save_pretrained(...)       – save to disk as a normal HF model

  At inference time the (quantized) model can be wrapped with
  torch.compile for further runtime speedup.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Track onnx (CPU / server)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Exports the model to ONNX, then applies ONNX Runtime dynamic INT8
  quantization.

  This is the gold standard for CPU / server deployment:
    – MatMul, Gemm, Conv weights are stored as INT8
    – INT8 kernel implementations in ORT are highly optimised for x86/ARM
    – Typically 2–4× faster than FP32 on CPU; ~50% smaller model file

  NOTE: OnnxRuntime does not use MPS (Apple GPU).  On an Apple Silicon
  machine this path runs on the CPU — still useful for comparing with a
  hypothetical server deployment, but expect it to be slower than the
  MPS-backed Track A on your local machine.

Usage
-----
    python -m src.models.quantize                        # both tracks
    python -m src.models.quantize --track quanto         # quanto only
    python -m src.models.quantize --track onnx           # onnx int8 only
    python -m src.models.quantize --model-dir models/ast_mixup

CLI flags
---------
  --model-dir   Source model directory   (default: models/ast_baseline)
  --track       quanto | onnx | both             (default: both)
"""

import argparse
import json
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process
from optimum.quanto import freeze, qint8, quantization_map, quantize
from transformers import AutoModelForAudioClassification

from src.models.export_onnx import export_to_onnx

DEFAULT_MODEL_DIR = "models/ast_baseline"
OUTPUT_QUANTO = "models/ast_quanto_int8"
OUTPUT_ONNX_FP32 = "models/ast_onnx"
OUTPUT_ONNX_INT8 = "models/ast_onnx_int8"


def quantize_quanto(model_dir: str, output_dir: str) -> None:
    """
    Quantise a HF AST model to INT8 weights using optimum.quanto.

    Compatible with MPS, CUDA, and CPU.  The saved model is a standard
    HF directory and can be loaded with AutoModelForAudioClassification.

    Parameters
    ----------
    model_dir  : Source model directory.
    output_dir : Destination directory for the quantised model.
    """

    print(f"\n{'─'*60}")
    print("optimum.quanto INT8 weight quantisation")
    print(f"{'─'*60}")
    print(f"  Source  : {model_dir}")
    print(f"  Output  : {output_dir}\n")
    print("Loading model …")
    model = AutoModelForAudioClassification.from_pretrained(model_dir)
    model.eval()

    print("Quantising weights to INT8 …")
    quantize(model, weights=qint8)

    print("Freezing (committing quantised weights) …")
    freeze(model)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)

    # save quantization map
    qmap_path = Path(output_dir) / "quantization_map.json"
    with open(qmap_path, "w") as fh:
        json.dump(quantization_map(model), fh, indent=2)

    print(f"\nQuantised model saved → '{output_dir}'")
    print("  Load with: AutoModelForAudioClassification.from_pretrained"
          f"('{output_dir}')")

    # quick sanity check
    _quanto_size = sum(p.numel() * p.element_size()
                       for p in model.parameters()) / 1e6
    print(f"  In-memory param size (approx): {_quanto_size:.1f} MB")


def quantize_onnx_int8(model_dir: str, onnx_fp32_dir: str,
                       onnx_int8_dir: str) -> None:
    """
    Export model to ONNX (float32) then apply ONNX Runtime dynamic INT8
    quantisation.

    "Dynamic" quantisation computes activation scales on-the-fly per
    inference — no calibration dataset is required.  MatMul / Gemm / Conv
    weight kernels are stored as INT8, cutting the model file roughly in
    half and unlocking INT8-optimised CPU kernels in ORT.

    Parameters
    ----------
    model_dir     : Source HF model directory.
    onnx_fp32_dir : Where to write the float32 ONNX model.
    onnx_int8_dir : Where to write the INT8 quantised ONNX model.
    """

    print(f"\n{'─'*60}")
    print("onnx + ort dynamic INT8 quantisation")
    print(f"{'─'*60}")
    print(f"  Source      : {model_dir}")
    print(f"  ONNX FP32   : {onnx_fp32_dir}")
    print(f"  ONNX INT8   : {onnx_int8_dir}\n")

    fp32_onnx_path = export_to_onnx(model_dir, onnx_fp32_dir)

    Path(onnx_int8_dir).mkdir(parents=True, exist_ok=True)
    int8_onnx_path = str(Path(onnx_int8_dir) / "model.onnx")

    # ORT's recommended pre-processing pass: propagates shapes, merges
    # duplicate initializers, and inserts missing symbolic dim values so that
    # quantize_dynamic's internal shape-inference step succeeds.
    preprocessed_path = str(Path(onnx_int8_dir) / "model_preprocessed.onnx")
    print("\nRunning ORT quant_pre_process (shape propagation) …")
    quant_pre_process(fp32_onnx_path, preprocessed_path)

    print(f"\nApplying ORT dynamic INT8 quantisation …")
    quantize_dynamic(
        model_input=preprocessed_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QInt8,
    )

    Path(preprocessed_path).unlink(missing_ok=True)

    print(f"INT8 ONNX model saved → '{int8_onnx_path}'")

    # size comparison
    fp32_mb = Path(fp32_onnx_path).stat().st_size / 1e6
    int8_mb = Path(int8_onnx_path).stat().st_size / 1e6
    print(f"\n  FP32 model : {fp32_mb:.1f} MB")
    print(f"  INT8 model : {int8_mb:.1f} MB  "
          f"({100 * (1 - int8_mb / fp32_mb):.0f}% smaller)")
    print("\n  Run inference with:")
    print("    import onnxruntime as ort")
    print(f"    sess = ort.InferenceSession('{int8_onnx_path}')")
    print("    logits = sess.run(['logits'], {'input_values': arr})[0]")


# cli

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="INT8 quantisation for the AST model (two tracks).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Path to saved model directory",
    )
    parser.add_argument(
        "--track",
        choices=["quanto", "onnx", "both"],
        default="both",
        help=("quanto = optimum.quanto (PyTorch/MPS), "
              "onnx = ONNX + ORT int8 (CPU/server), "
              "both = run both tracks"),
    )
    args = parser.parse_args()

    if args.track in ("quanto", "both"):
        quantize_quanto(
            model_dir=args.model_dir,
            output_dir=OUTPUT_QUANTO,
        )

    if args.track in ("onnx", "both"):
        quantize_onnx_int8(
            model_dir=args.model_dir,
            onnx_fp32_dir=OUTPUT_ONNX_FP32,
            onnx_int8_dir=OUTPUT_ONNX_INT8,
        )
