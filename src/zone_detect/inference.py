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

    onnx_inputs = imgs.numpy()

    input_name = ort_session.get_inputs()[0].name

    onnxruntime_input = {input_name: onnx_inputs}

    ########## Trying io binding
    """
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name
    output_shape = ort_session.get_outputs()[0].shape
    batch_size, channels, height, width = imgs.shape

    # Handle dynamic output shape if necessary
    if output_shape is None or any(dim is None for dim in output_shape):
        output_shape = (
            batch_size,
            ort_session.get_outputs()[0].type.shape[1],
            height,
            width,
        )

    input_ortvalue = ort.OrtValue.ortvalue_from_numpy(onnx_inputs, "cuda", 0)
    output_ortvalue = ort.OrtValue.ortvalue_from_shape_and_type(
        output_shape,
        np.float32,
        "cuda",
        0,
    )

    ort_session_io = ort_session.io_binding()

    ort_session_io.bind_input(
        name=input_name,
        device_type=input_ortvalue.device_name(),
        device_id=0,
        element_type=np.float32,
        shape=input_ortvalue.shape(),
        buffer_ptr=input_ortvalue.data_ptr(),
    )

    ort_session_io.bind_output(
        name=output_name,
        device_type=output_ortvalue.device_name(),
        device_id=0,
        element_type=np.float32,
        shape=output_ortvalue.shape(),
        buffer_ptr=output_ortvalue.data_ptr(),
    )

    ort_session.run_with_iobinding(ort_session_io)
    logits = ort_session_io.copy_outputs_to_cpu()[0]
    """
    ##########

    # sane, default way
    logits = ort_session.run(None, onnxruntime_input)[0]
    predictions = softmax(logits, axis=1)

    return predictions, samples["index"].numpy()
