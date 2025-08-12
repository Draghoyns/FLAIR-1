import torch
from optimum.quanto import quantize, qint8, Calibration, freeze
from tqdm import tqdm

from src.zone_detect.optimization.calibration import load_calibration_images


def with_quanto(model: torch.nn.Module, quant_args: dict) -> torch.nn.Module:
    """
    This function is a placeholder for the quanto quantization method.
    It is not implemented yet and serves as a reminder to implement it in the future.
    This method of quantization is compatible with torch.compile
    """

    precision, weights, activations, calibration, calibration_path = (
        quant_args.get("precision", "qint8"),
        quant_args.get("weights", "qint8"),
        quant_args.get("activations", "qint8"),
        quant_args.get("calibration", False),
        quant_args.get("calibration_dataset", ""),
    )

    quantize(model, weights=weights, activations=activations)

    if calibration and calibration_path:
        samples_list = load_calibration_images(calibration_path)
        with Calibration(momentum=0.5):
            for batch in tqdm(samples_list, desc="Calibrating model with batches..."):
                model(batch)

                del batch
                torch.cuda.empty_cache()
        # this is done on the fly !!!

    else:
        print("Calibration not enabled or path not provided, skipping calibration.")

    freeze(model)

    return model


def with_torchao(model: torch.nn.Module, quant_args: dict) -> torch.nn.Module:
    """
    This function is a placeholder for the torchao quantization method.
    It is not implemented yet and serves as a reminder to implement it in the future.
    """
    print("Quantization not implemented for this dtype, this is a placeholder.")
    return model


def with_pytorch(model: torch.nn.Module, quant_args: dict) -> torch.nn.Module:
    """
    This function is a placeholder for the torch quantization method.
    It is not implemented yet and serves as a reminder to implement it in the future.
    """
    print("Quantization not implemented for this dtype, this is a placeholder.")
    return model
