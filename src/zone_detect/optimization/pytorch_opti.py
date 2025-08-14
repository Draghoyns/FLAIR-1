import yaml
import os

from typing import Any
import torch
from safetensors.torch import save_file

from src.zone_detect.optimization.pruning import opti_pruna, sparsity
from src.zone_detect.optimization.quantization.quant_methods import (
    with_quanto,
    with_torchao,
    with_pytorch,
)


def pt_optimize_model(
    config: dict, model: torch.nn.Module, verbose: bool = False
) -> torch.nn.Module:
    """Optimize a PyTorch model for inference.
    Available optimizations are pruning, quantization and compilation."""

    opti_config = load_opti_config(config)

    if opti_config.get("prune", False):
        prune_params = opti_config.get("prune_args", {})
        model = opti_pruning(model, prune_params, verbose)

    if opti_config.get("quantize", False):
        quant_method = opti_config.get("quantize_method", "pytorch")
        quant_args = opti_config.get(f"{quant_method}_args", {})

        model = opti_quantization(model, quant_args, verbose)

    # save model if switch in config
    elif config.get("save_opti_model", False):
        model_out_path = config.get("output_path", "") / "model.safetensors"
        save_file(model.state_dict(), model_out_path)

    if opti_config.get("compile", False):
        model = opti_compile(model, verbose)

    return model.eval()


def opti_pruning(
    model: torch.nn.Module, params: dict, verbose: bool = False
) -> torch.nn.Module:
    """Apply pruning to the model using pruna ai."""

    model = opti_pruna(model, params)

    if verbose:
        sparsity(model)

    return model


def opti_quantization(
    model: torch.nn.Module, quant_args: dict, verbose: bool = False
) -> torch.nn.Module:
    """Apply quantization to the model."""

    dtype = getattr(torch, quant_args.get("precision", "float32"), torch.float32)

    if verbose:
        print(f"Quantizing model to {dtype}...")

    if dtype == torch.float32:
        return model
    if "float16" in str(dtype):
        # simple truncation
        model = model.to(dtype)
        # converting all parameters to bfloat16
        for param in model.parameters():
            param.requires_grad = False

        original_forward = model.forward

        def new_forward(*args: Any, **kwargs: Any) -> Any:
            args = tuple(arg.to(dtype) if hasattr(arg, "to") else arg for arg in args)
            kwargs = {
                k: v.to(dtype) if hasattr(v, "to") else v for k, v in kwargs.items()
            }
            return original_forward(*args, **kwargs)

        model.forward = new_forward

        # save model if switch in config

    else:
        # dtype is real quantization, not 16 bit

        method = quant_args.get("flag", "pytorch")
        precision = quant_args.get("precision", "float32")

        quant_function = eval(f"with_{method}")
        # en vrai faut pas faire ça c'est joli mais pas robuste

        if quant_function is None:
            raise ValueError(f"Quantization method '{method}' is not implemented.")

        model = quant_function(model, quant_args)

        # use provided quantization method
        # check precision and device compatibility
        # apply quantization
        # save with specificities if switch in config

        print(
            "Quantization not fully implemented for this dtype, this is a placeholder."
        )
        pass

    return model


def opti_compile(model: torch.nn.Module, verbose: bool = False):
    """Compile the PyTorch model for optimization.
    Compilation is done in-place."""

    if verbose:
        print(f"Compiling model...")

    model.compile(mode="reduce-overhead")

    return model


def load_opti_config(config: dict, verbose: bool = False) -> dict:
    """Load the optimization configuration from a YAML file."""

    opti_path = config.get("opti_config")

    if not opti_path or not os.path.exists(opti_path):
        if verbose:
            print(f"Optimization config file not found -- falling back to default.")

        opti_config = {
            "compile": False,
            "prune": False,
            "quantize": False,
        }
    else:
        with open(opti_path, "r") as f:
            opti_config = yaml.safe_load(f)

    return opti_config
