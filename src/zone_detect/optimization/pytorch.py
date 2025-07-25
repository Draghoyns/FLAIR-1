import torch

from src.zone_detect.metrics.metrics import sparsity
from src.zone_detect.model import opti_pruna


def pt_optimize_model(
    config: dict, model: torch.nn.Module, verbose: bool = False
) -> torch.nn.Module:
    """Optimize a PyTorch model for inference.
    Available optimizations are pruning, quantization and compilation."""

    pruna_flag = config.get("pruna", False)
    pruna_params = config.get("pruna_args", {})

    quant_precision = config.get("precision", "float32")
    dtype = getattr(torch, quant_precision, torch.float32)

    compile_flag = config.get("compile", False)

    if pruna_flag:
        model = opti_pruning(model, pruna_params, verbose)

    if quant_precision != "float32" and torch.cuda.is_available():
        model = opti_quantization(model, dtype, verbose)

    if compile_flag:
        model = opti_compile(model, verbose)

    return model.eval()


def opti_pruning(
    model: torch.nn.Module, params: dict, verbose: bool = False
) -> torch.nn.Module:
    """Apply pruning to the model using pruna ai."""

    model = opti_pruna(model, params)

    if verbose:
        sparsity({"model": model})

    return model


def opti_quantization(
    model: torch.nn.Module, dtype: torch.dtype, verbose: bool = False
) -> torch.nn.Module:
    """Apply quantization to the model."""

    if verbose:
        print(f"Quantizing model to {dtype}...")

    if dtype == torch.bfloat16:
        # simple truncation
        model = model.to(dtype)

    else:
        # apply quantization
        print("Quantization not implemented for this dtype, this is a placeholder.")
        pass

    return model


def opti_compile(model: torch.nn.Module, verbose: bool = False):
    """Compile the PyTorch model for optimization.
    Compilation is done in-place."""

    if verbose:
        print(f"Compiling model...")

    model.compile(mode="reduce-overhead")

    return model
