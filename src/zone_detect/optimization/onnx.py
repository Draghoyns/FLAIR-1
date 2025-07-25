import onnxruntime as ort
from pathlib import Path


def onnx_optimize_model(config: dict, onnx_path: Path) -> Path:
    """Takes an ONNX model file and quantizes it if necessary."""

    if config.get("precision", "fp32") != "fp32":
        # do quantization -> check quantization module oopsie
        print("Quantization is not implemented yet, it's a placeholder for logic.")
        return onnx_path
    else:
        return onnx_path


def get_session(config: dict, onnx_path: Path) -> ort.InferenceSession:
    """Create an ONNX InferenceSession."""
    device_key = "gpu" if config["use_gpu"] else "cpu"
    providers = {"gpu": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider"}

    return ort.InferenceSession(onnx_path, providers=[providers[device_key]])
