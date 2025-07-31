from typing import Any
import torch
import torch.nn as nn
import os

from src.zone_detect.model import load_model
from src.zone_detect.utils import read_config


def dummy(model_handler: Any) -> tuple[torch.Tensor]:

    cfg = model_handler.model_path
    config = read_config({"conf": cfg})

    batch_size = config.get("batch_size", 2)
    patch_size = config.get("img_pixels_detection", 512)  # default patch size
    n_bands = len(config["channels"]) if "channels" in config else 3

    dummy_input = (torch.randn(batch_size, n_bands, patch_size, patch_size),)

    return dummy_input


def load_model_from_cfg_path(cfg_path: str) -> nn.Module:
    """
    Load a model from a givne file.
    Args:
        cfg_path (str): Path to the configuration file.
    Returns:
        nn.Module: The loaded model.
    """

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file {cfg_path} does not exist.")
    config = read_config({"conf": cfg_path})

    return load_model(config)
