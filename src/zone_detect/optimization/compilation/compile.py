import torch


def compile_model(
    model: torch.nn.Module,
):
    """Compile the PyTorch model for optimization.
    Compilation is done in-place."""

    model.compile(mode="reduce-overhead")

    return model


# compile
# warmup
# inference
