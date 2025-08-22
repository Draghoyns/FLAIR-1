import torch
from tqdm import tqdm

from optimum.quanto import quantize, qint8, Calibration, freeze
from torchao.quantization import int8_weight_only, quantize_

from src.zone_detect.optimization.calibration import load_calibration_images


def with_quanto(model: torch.nn.Module, quant_args: dict) -> torch.nn.Module:
    """
    This function is for the quanto quantization method.
    This method of quantization is incompatible with torch.compile
    """

    method = quant_args.get("method", "dynamic")
    method_args = quant_args.get("methods", {}).get(method, {})

    weights, activations, calibration, calibration_path = (
        eval(method_args.get("weights", qint8)),
        method_args.get("activations", None),
        method_args.get("calibration", False),
        quant_args.get("calibration_dataset", ""),
    )
    if activations:
        activations = eval(activations)

    quantize(model, weights=weights, activations=activations)
    device = next(model.parameters()).device

    if calibration and calibration_path:

        samples_list = load_calibration_images(calibration_path, device=device)
        with Calibration(momentum=0.9):
            with torch.no_grad():
                for batch in tqdm(
                    samples_list, desc="Calibrating model with batches..."
                ):
                    batch = batch.to(device=device)
                    model(batch)

                    del batch
                    torch.cuda.empty_cache()
            # this is done on the fly !!!:
    else:
        print(
            "No calibration dataset provided."
            "This may lead to suboptimal quantization results."
        )
    freeze(model)

    return model


def with_torchao(model: torch.nn.Module, quant_args: dict) -> torch.nn.Module:
    """
    This function is a placeholder for the torchao quantization method.
    It is not implemented yet and serves as a reminder to implement it in the future.
    """
    precision = quant_args.get("precision", "float32")

    if precision.startswith("int8"):
        model = model.eval()
        model = model.to(torch.bfloat16)
        quantize_(model, int8_weight_only(group_size=32))  # type: ignore

    return model


def with_pytorch(model: torch.nn.Module, quant_args: dict) -> torch.nn.Module:
    """
    This function is a placeholder for the torch quantization method.
    It is not implemented yet and serves as a reminder to implement it in the future.
    """
    print("Quantization not implemented for this dtype, this is a placeholder.")
    return model
