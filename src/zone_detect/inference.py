import numpy as np
from scipy.special import softmax
import torch

from typing import Any

import onnxruntime as ort


def inference(
    model_type: str,
    config: dict[str, Any],
    args: dict[str, Any],
    samples: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:

    if model_type == "pytorch":
        return inference_pt(
            device=args["device"],
            model=args["model"],
            use_gpu=args["use_gpu"],
            config=config,
            samples=samples,
        )
    elif model_type == "onnx":
        return inference_onnx(ort_session=args["ort_session"], samples=samples)
    else:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported types are 'pytorch' and 'onnx'."
        )


def inference_pt(
    device: torch.device,
    model: torch.nn.Module,
    use_gpu: bool,
    config: dict[str, Any],
    samples: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    imgs = samples["image"].to(device, non_blocking=(device.type == "cuda"))
    if use_gpu:
        torch.cuda.synchronize()
    with torch.no_grad():
        logits = model(imgs)
        if config.get("model_framework", {}).get("model_provider") == "HuggingFace":
            logits = logits.logits
        logits.to(device)
    predictions = torch.softmax(logits, dim=1)
    predictions = predictions.cpu().numpy()
    indices = samples["index"].cpu().numpy()

    return predictions, indices


def inference_onnx(
    ort_session: ort.InferenceSession,
    samples: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:

    imgs = samples["image"]

    # onnx_inputs = [np.expand_dims(tensor.numpy(), axis=0) for tensor in imgs]
    onnx_inputs = imgs.numpy()

    onnxruntime_input = {ort_session.get_inputs()[0].name: onnx_inputs}

    logits = ort_session.run(None, onnxruntime_input)[0]
    predictions = softmax(logits, axis=1)

    return predictions, samples["index"].numpy()
