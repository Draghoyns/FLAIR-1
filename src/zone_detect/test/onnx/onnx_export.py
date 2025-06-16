import numpy as np
from pathlib import Path
from typing import Any

import rasterio
from rasterio.enums import Resampling
import rasterio.windows
from typing import Optional

from scipy.special import softmax

import torch
from torch.onnx import export, ONNXProgram
import onnx
import onnxruntime
from onnxsim import simplify

from src.zone_detect.model import load_model


#### Swin IRC model ####
# transfomers


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


def export_onnx(config: dict[str, Any], out_name: str = "opti"):
    """
    Export a HuggingFace model to ONNX format.
    """
    save_directory = Path(config["model_weights"]).parent

    model_name = config["model_framework"]["HuggingFace"]["org_model"]
    filename = f"{model_name}_{out_name}.onnx" if out_name else f"{model_name}.onnx"
    output_path = save_directory / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    patch_size = config.get(
        "img_pixels_detection", 512
    )  # do we really need to ? can't it be dynamic ?
    n_bands = len(config["channels"])
    batch_size = config.get("batch_size", 1)

    model = load_model(config)
    model.eval()

    assert (
        config["model_framework"]["model_provider"] == "HuggingFace"
    ), "Model provider must be HuggingFace for ONNX export."

    dummy_input = (torch.randn(batch_size, n_bands, patch_size, patch_size),)

    with torch.no_grad():
        onnx_export = export(
            model,
            dummy_input,
            do_constant_folding=True,
            dynamo=True,
        )

        if onnx_export is not None:
            if out_name == "opti":
                onnx_export.optimize()

            # simplify with ORT
            simp_onnx_path = simplify_onnx(output_path, onnx_export)

            print(f"Model exported to ONNX format to {output_path}")
            check_export_onnx(simp_onnx_path)
        else:
            print("ONNX export failed.")
            simp_onnx_path = output_path

    return simp_onnx_path


def get_onnx_path(config: dict[str, Any]) -> Path:

    model_name = config["model_framework"]["HuggingFace"]["org_model"]
    model_names = model_name.split("/")  # Get the last part of the model name
    # e.g. "openmmlab/upernet-swin-small" -> "upernet-swin-small"
    onnx_path = (
        Path(config["model_weights"]).parent
        / model_names[0]
        / f"simplified_{model_names[-1]}_opti.onnx"
    )

    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}. Exporting...")
        onnx_path = export_onnx(config)

    return onnx_path


#### ONNX inference ####
# _____________________#


def inference_onnx(
    ort_session: onnxruntime.InferenceSession, img: torch.Tensor
) -> np.ndarray:
    """Perform inference using ONNX Runtime.
    Args:
        model_path (Path): Path to the ONNX model.
        img (torch.Tensor): Input image tensor, first dimension is batch size."""

    # for the test device is cpu by default

    onnx_inputs = [np.expand_dims(tensor.numpy(), axis=0) for tensor in img]
    # CHatGPT's suggestion : onnx_inputs = img.numpy()
    # print(f"Input tensor shape for ONNX: {onnx_inputs[0].shape}")
    # print(f"ONNX inputs: {onnx_inputs}")

    # to move outside maybe

    onnxruntime_input = {
        input_arg.name: input_value
        for input_arg, input_value in zip(ort_session.get_inputs(), onnx_inputs)
    }

    # ONNX Runtime returns a list of outputs
    logits = ort_session.run(None, onnxruntime_input)[0]
    predictions = softmax(logits, axis=1)

    return predictions


def dummy_input_onnx(input_img: Path):
    """Returns a dummy input tensor of shape (batch_size (1 here), num_bands, patch_size, patch_size)"""

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

    ort_session = onnxruntime.InferenceSession(onnx_model_path)

    onnx_outputs = inference_onnx(ort_session, input_tensor)

    """
    print(f"PyTorch output type: {type(predictions)}")
    print(f"ONNX output type: {type(onnx_outputs)}")
    print(f"PyTorch output shape: {predictions.shape}")
    print(f"ONNX output shape: {onnx_outputs.shape}")
    """

    rtol, atol = 1e-7, 1e-7
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


#### ONNX simplification ####
# ___________________________#


def patch_constant_nodes(model: onnx.ModelProto) -> onnx.ModelProto:
    new_nodes = []

    for node in model.graph.node:
        if node.op_type == "Constant":
            attr_names = {attr.name for attr in node.attribute}

            if "value_ints" in attr_names and "value" not in attr_names:
                for attr in node.attribute:
                    if attr.name == "value_ints":
                        tensor = onnx.helper.make_tensor(
                            name=node.output[0],
                            data_type=onnx.TensorProto.INT64,
                            dims=[len(attr.ints)],
                            vals=attr.ints,
                        )
                        new_node = onnx.helper.make_node(
                            "Constant",
                            inputs=[],
                            outputs=node.output,
                            name=node.name,
                            value=tensor,
                        )
                        new_nodes.append(new_node)
                        break
            else:
                new_nodes.append(node)
        else:
            new_nodes.append(node)

    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)
    print(f"✅ Patched model")

    return model


def simplify_onnx(onnx_path: Path, onnx_program: Optional[ONNXProgram] = None) -> Path:
    """
    Simplify the ONNX model using onnx-simplifier
    Returns the simplified model, which can be used as a standard ONNX model object.
    """

    # convert model
    if onnx_program is None:
        model = onnx.load(onnx_path)
    else:
        model = onnx_program.model_proto

    # fix missing value attribute in Constant nodes
    model = patch_constant_nodes(model)

    # double check
    # debug_onnx_model(onnx_path)

    model_simp, check = simplify(model)

    assert check, "Simplified ONNX model could not be validated"

    # save the simplified model
    simplified_path = onnx_path.parent / f"simplified_{onnx_path.name}"

    onnx.save(model_simp, simplified_path)
    print(f"ONNX model simplified and saved to {simplified_path}")

    return simplified_path


def debug_onnx_model(onnx_path: Path):
    model = onnx.load(str(onnx_path))
    for i, node in enumerate(model.graph.node):
        if node.op_type == "Constant":
            has_value = any(attr.name == "value" for attr in node.attribute)
            if not has_value:
                print(f"\nNode {i}: {node.op_type}")
            for attr in node.attribute:
                if not has_value:
                    print(f" - {attr.name} (type: {attr.type})")
                    print(f"❗ Missing 'value' attribute in {node.name} node {i}")


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
    out_opti = "opti"
    out_simple = "not-opti"
    onnx_opti = export_onnx(config, out_name=out_opti)
    # onnx_simple = export_onnx_hf(config, Path(save_directory), out_name=out_simple)

    input_img = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/ortho/D037_2021/UU_S1_4/037_2021_UU_S1_4_IRC.tif"

    dummy = dummy_input_onnx(Path(input_img))

    # print(f"Dummy input shape: {dummy.shape}")

    """
    pred_simple = inference_onnx(Path(onnx_simple), dummy)
    pred_opti = inference_onnx(Path(onnx_opti), dummy)

    mse = np.mean((pred_simple - pred_opti) ** 2)
    print(f"Mean Squared Error between simple and optimized ONNX predictions: {mse}")

    print("Inference completed successfully.")

    print("Comparing ONNX optimized model with PyTorch model...")
    compare_to_pytorch(config, Path(onnx_opti), dummy)
    """

    # Simplify the ONNX model
    print("Simplifying the ONNX model...")
    simplified_onnx_opti_path = simplify_onnx(onnx_opti)

    print("Comparing simplified ONNX model with PyTorch model...")
    compare_to_pytorch(config, simplified_onnx_opti_path, dummy)
