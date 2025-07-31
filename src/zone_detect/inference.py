import numpy as np
from scipy.special import softmax
import torch

from typing import Any

import onnxruntime as ort
import tqdm


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
            quant_type=args.get("dtype", torch.float32),
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
    quant_type: torch.dtype = torch.float32,
) -> tuple[np.ndarray, np.ndarray]:
    imgs = samples["image"].to(device, non_blocking=(device.type == "cuda"))
    if use_gpu:
        torch.cuda.synchronize()
    with torch.no_grad():
        if quant_type != torch.float32 and use_gpu:
            imgs = imgs.to(quant_type)
        logits = model(imgs)
        if config.get("model_framework", {}).get("model_provider") == "HuggingFace":
            logits = logits.logits
        logits.to(device)
    predictions = torch.softmax(logits, dim=1)
    predictions = predictions.to(torch.float32).cpu().numpy()
    indices = samples["index"].cpu().numpy()

    return predictions, indices


def inference_onnx(
    ort_session: ort.InferenceSession,
    samples: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:

    imgs = samples["image"]

    # ensure batch size is sound
    original_batch_size = imgs.shape[0]
    expected_batch_size = ort_session.get_inputs()[0].shape[0]

    if original_batch_size < expected_batch_size:
        padding_count = expected_batch_size - original_batch_size
        imgs = torch.cat([imgs, imgs[-1:].repeat(padding_count, 1, 1, 1)], dim=0)
    # ___________________________

    onnx_inputs = imgs.numpy()

    input_name = ort_session.get_inputs()[0].name

    onnxruntime_input = {input_name: onnx_inputs}

    logits = ort_session.run(None, onnxruntime_input)[0]
    predictions = softmax(logits, axis=1)

    return predictions[:original_batch_size], samples["index"].numpy()


def warmup(
    model_type: str,
    config: dict[str, Any],
    args: dict[str, Any],
) -> None:

    size = (
        config["batch_size"],
        len(config["channels"]),
        config["img_pixels_detection"],
        config["img_pixels_detection"],
    )
    samples = {"image": torch.randn(*size), "index": torch.tensor([0])}

    for _ in tqdm.trange(10, desc="Warming up model"):
        _ = inference(
            model_type=model_type,
            config=config,
            args=args,
            samples=samples,
        )
