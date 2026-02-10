"""
Export a HuggingFace model from models/ to ONNX with optional dynamic quantization.
Run from project root with venv activated:
  python tools/export_onnx.py --model-dir models/YELP-Review_Classifier [--quantize]
"""
import argparse
import sys
from pathlib import Path

# Project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def export_onnx(model_dir: Path, output_path: Path, opset: int = 14) -> Path:
    """Export HuggingFace model to ONNX (no quantization)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = Path(model_dir).resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()

    # Dummy inputs for tracing
    dummy = tokenizer(
        "dummy input",
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True,
    )
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "attention_mask": {0: "batch_size"},
        "logits": {0: "batch_size"},
    }

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy["input_ids"], dummy["attention_mask"]),
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    return output_path


def quantize_onnx(onnx_path: Path, output_path: Path) -> Path:
    """Apply ONNX Runtime dynamic quantization to reduce size and improve speed."""
    from onnxruntime.quantization import quantize_dynamic

    onnx_path = Path(onnx_path).resolve()
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        str(onnx_path),
        str(output_path),
        weight_type=1,  # QInt8
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export HuggingFace model to ONNX (optional quantization).")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "models" / "YELP-Review_Classifier",
        help="Path to model directory under models/",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX path (default: <model_dir>/model.onnx)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Apply dynamic quantization (output: model_quantized.onnx)",
    )
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()

    model_dir = args.model_dir if args.model_dir.is_absolute() else ROOT / args.model_dir
    output = args.output or (model_dir / "model.onnx")

    print(f"Exporting {model_dir} to ONNX...")
    export_onnx(model_dir, output, opset=args.opset)
    print(f"Saved: {output}")

    if args.quantize:
        quantized = output.parent / "model_quantized.onnx"
        print("Quantizing...")
        quantize_onnx(output, quantized)
        print(f"Saved: {quantized}")


if __name__ == "__main__":
    main()
