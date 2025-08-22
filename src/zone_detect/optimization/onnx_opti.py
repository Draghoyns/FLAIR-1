from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process


def onnx_optimize_model(config: dict, onnx_path: Path) -> Path:
    """Takes an ONNX model file and quantizes it if necessary."""

    if "float32" not in config.get("onnx_quant_precision", "float32"):
        # int8 by default honestly

        # preprocessing
        out_preprocessed = (
            Path(onnx_path).parent / f"{onnx_path.stem}_preprocessed.onnx"
        )
        out_quant = Path(onnx_path).parent / f"{onnx_path.stem}_quantized.onnx"
        quant_pre_process(onnx_path, out_preprocessed)

        quantize_dynamic(
            out_preprocessed,
            out_quant,
        )

        print("Quantized model saved to:", out_quant)
        return out_quant
    else:

        return onnx_path


def get_session(config: dict, onnx_path: Path) -> ort.InferenceSession:
    """Create an ONNX InferenceSession."""
    device_key = "gpu" if config["use_gpu"] else "cpu"
    providers = {"gpu": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider"}

    return ort.InferenceSession(onnx_path, providers=[providers[device_key]])
