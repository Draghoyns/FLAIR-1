import numpy as np
from pathlib import Path
from typing import Any

import rasterio
from rasterio.enums import Resampling
import rasterio.windows

from scipy.special import softmax

import torch
from torch.onnx import export
import onnx
import onnxruntime
from onnxsim import simplify

from src.zone_detect.test.onnx.model_loading import load_model


#### Swin IRC model ####
# transfomers


def export_onnx_hf(config: dict[str, Any], save_directory: Path):
    """
    Export a HuggingFace model to ONNX format.
    """

    model_name = config["model_framework"]["HuggingFace"]["org_model"]
    onnx_dir = f"{save_directory}/onnx_export"
    output_path = Path(onnx_dir) / "upernet_swin_dynamic_opti.onnx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    patch_size = config.get("patch_size", 512)

    print(f"Exporting {model_name} to ONNX format...")

    model = load_model(config)
    model.eval()

    assert (
        config["model_framework"]["model_provider"] == "HuggingFace"
    ), "Model provider must be HuggingFace for ONNX export."

    dummy_input = (torch.randn(1, 3, patch_size, patch_size),)

    with torch.no_grad():
        onnx_export = export(
            model,
            dummy_input,
            dynamo=True,
        )

        if onnx_export is not None:
            onnx_export.optimize()

            onnx_export.save(output_path)
            print(f"Model exported to {output_path}")
        else:
            print("ONNX export failed.")

    return output_path


def check_export_onnx(onnx_path: Path):
    """
    Check if the ONNX export is successful.
    """
    if onnx_path is None:
        print("No ONNX file found.")
    else:
        print(f"ONNX file found")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model is well formed, whatever that means.")


def inference_onnx(model_path: Path, img: torch.Tensor) -> np.ndarray:
    """Perform inference using ONNX Runtime.
    Args:
        model_path (Path): Path to the ONNX model.
        img (torch.Tensor): Input image tensor, first dimension is batch size."""

    # for the test device is cpu by default

    onnx_inputs = [np.expand_dims(tensor.numpy(), axis=0) for tensor in img]
    # CHatGPT's suggestion : onnx_inputs = img.numpy()
    print(f"Input tensor shape for ONNX: {onnx_inputs[0].shape}")
    # print(f"ONNX inputs: {onnx_inputs}")

    # to move outside maybe
    ort_session = onnxruntime.InferenceSession(
        model_path, providers=["CPUExecutionProvider"]
    )

    onnxruntime_input = {
        input_arg.name: input_value
        for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)
    }

    # ONNX Runtime returns a list of outputs
    logits = ort_session.run(None, onnxruntime_input)[0]
    predictions = softmax(logits, axis=1)

    return predictions


def dummy_input_onnx(input_img: Path):
    # fixed parameters for test purposes
    # test on the first tile
    bands = [1, 2, 3]
    num_bands = len(bands)
    patch_size = 512
    margin = 0
    norm_means = [106.38, 105.08, 110.87]
    norm_stds = [39.69, 52.17, 45.38]

    src = rasterio.open(input_img)
    min_x, min_y, max_x, max_y = src.bounds
    resolution_x, resolution_y = map(lambda r: abs(round(r, 5)), src.res)
    bounds = (
        min_x - margin * resolution_x,
        min_x + (patch_size - margin) * resolution_x,
        min_y - margin * resolution_y,
        min_y + (patch_size - margin) * resolution_y,
    )

    window = rasterio.windows.from_bounds(*bounds, transform=src.meta["transform"])
    patch_img = src.read(
        indexes=bands,
        window=window,
        out_shape=(num_bands, patch_size, patch_size),
        resampling=Resampling.bilinear,
        boundless=True,
    )

    # Normalization

    img = patch_img.astype(np.float64)
    for i in range(num_bands):
        img[i] = (img[i] - norm_means[i]) / norm_stds[i]

    img = np.expand_dims(img, axis=0)  # Add batch dimension

    return torch.as_tensor(img, dtype=torch.float)


def compare_to_pytorch(
    config: dict[str, Any], onnx_model_path: Path, input_tensor: torch.Tensor
) -> None:
    """
    Compare the output of a PyTorch model with an ONNX model.
    """
    pt_model = load_model(config)
    pt_model.eval()
    with torch.no_grad():
        pt_output = pt_model(input_tensor)
        logits = pt_output.logits
    predictions = torch.softmax(logits, dim=1)
    predictions = predictions.numpy()

    onnx_outputs = inference_onnx(onnx_model_path, input_tensor)

    """
    print(f"PyTorch output type: {type(predictions)}")
    print(f"ONNX output type: {type(onnx_outputs)}")
    print(f"PyTorch output shape: {predictions.shape}")
    print(f"ONNX output shape: {onnx_outputs.shape}")
    """

    rtol, atol = 1e-6, 1e-6
    for torch_output, onnxruntime_output in zip(predictions, onnx_outputs):
        np.testing.assert_allclose(
            onnxruntime_output, torch_output, atol=atol, rtol=rtol
        )

    print(
        f"PyTorch and ONNX Runtime output matched at relative tolerance {rtol} and absolute tolerance {atol}."
    )

    """
    print(f"Output length: {len(onnx_outputs)}")
    print(f"Sample output: {onnx_outputs}")
    """

    # plot error using imshow
    import matplotlib.pyplot as plt

    onnx_name = onnx_model_path.name

    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(np.mean(np.abs(predictions[0] - onnx_outputs[0]), axis=0), cmap="plasma")
    plt.title("Difference between PyTorch and ONNX outputs")
    plt.colorbar(label="Absolute difference")
    plt.title(
        f"Difference between PyTorch and ONNX {onnx_name} outputs (max diff: {np.max(np.abs(predictions[0] - onnx_outputs[0])):.4f})"
    )
    plt.savefig(
        f"{onnx_model_path.parent}/comparison_{onnx_name}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()  # Close the plot to free memory
    print(f"Comparison plot saved as comparison_{onnx_name}.png")

    # save plot


def simplify_onnx(onnx_path: Path) -> onnx.ModelProto:
    """
    Simplify the ONNX model using onnx-simplifier
    Returns the simplified model, which can be used as a standard ONNX model object.
    """

    # load your predefined ONNX model
    model = onnx.load(onnx_path)

    # convert model
    model_simp, check = simplify(model)

    assert check, "Simplified ONNX model could not be validated"

    return model_simp


#### ResNet-UNet RVB model ###
# convolution ?

if __name__ == "__main__":

    save_directory = "/home/ign.fr/SHys/FLAIR-1/src/zone_detect/test/onnx"

    model_name = "openmmlab/upernet-swin-small"
    model_ckpt = "/media/DATA/INFERENCE_HS/MODELS_IA/FLAIR1/swin-upernet-small_IRV_SET1/checkpoints/ckpt-epoch=84-val_loss=0.37_00_HF_SwinUpernet_Small_IR-R-G_set1.ckpt"

    config = {
        "model_framework": {
            "model_provider": "HuggingFace",
            "HuggingFace": {
                "org_model": model_name,
            },
        },
        "n_classes": 19,
        "channels": [1, 2, 3],
        "model_weights": model_ckpt,
        "patch_size": 512,
    }

    # Export the HuggingFace model to ONNX
    # out = export_onnx_hf(config, Path(save_directory))

    # check_export_onnx(out)

    input_img = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/ortho/D037_2021/UU_S1_4/037_2021_UU_S1_4_IRC.tif"
    onnx_simple = "/home/ign.fr/SHys/FLAIR-1/src/zone_detect/test/onnx/onnx_export/upernet_swin_dynamic_not-opti.onnx"
    onnx_opti = "/home/ign.fr/SHys/FLAIR-1/src/zone_detect/test/onnx/onnx_export/upernet_swin_dynamic_opti.onnx"

    dummy = dummy_input_onnx(Path(input_img))

    # print(f"Dummy input shape: {dummy.shape}")

    '''
    pred_simple = inference_onnx(Path(onnx_simple), dummy)
    pred_opti = inference_onnx(Path(onnx_opti), dummy)

    print(
        f"""Predicted shape :
    {pred_simple.shape}"""
    )

    mse = np.mean((pred_simple - pred_opti) ** 2)
    print(f"Mean Squared Error between simple and optimized ONNX predictions: {mse}")

    print("Inference completed successfully.")
    '''

    # Compare ONNX output with PyTorch model output
    print("Comparing ONNX simple model with PyTorch model...")
    compare_to_pytorch(config, Path(onnx_simple), dummy)

    print("Comparing ONNX optimized model with PyTorch model...")
    compare_to_pytorch(
        config, Path(onnx_opti), dummy
    )  # Use the optimized ONNX model for comparison
    print("Comparison completed successfully.")
