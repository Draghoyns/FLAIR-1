import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

import segmentation_models_pytorch as smp
from transformers import AutoModelForSemanticSegmentation, AutoConfig

from src.zone_detect.inference import warmup
from src.zone_detect.optimization.onnx_opti import get_session, onnx_optimize_model
from src.zone_detect.optimization.pytorch_opti import pt_optimize_model
from src.zone_detect.test.onnx_tests.onnx_export import get_onnx_path
from src.zone_detect.utils import read_config

Config = dict[str, Any]


@dataclass
class FLAIR_ModelFactory:
    """
    A factory class for creating models based on the provided configuration.
    This class supports models from both SegmentationModelsPytorch and HuggingFace.
    Arguments:
        config (Mapping): A dictionary containing the model configuration, with at least
                          the keys 'model_framework', 'model_provider', 'channels', and 'n_classes'.
                          For SegmentationModelsPytorch, it should include 'encoder_decoder'.
                          For HuggingFace, it should include 'org_model'.
    """

    config: Mapping
    model_provider: str = field(init=False)
    seg_model: nn.Module = field(init=False)

    def __post_init__(self):

        self.model_provider = self.config["model_framework"]["model_provider"]

        n_channels = int(len(self.config["channels"]))
        n_classes = self.config["n_classes"]

        if self.model_provider == "SegmentationModelsPytorch":
            encoder, architecture = self.config["model_framework"][
                "SegmentationModelsPytorch"
            ]["encoder_decoder"].split("_")
            self.seg_model = smp.create_model(
                arch=architecture,
                encoder_name=encoder,
                classes=n_classes,
                in_channels=n_channels,
            )

        elif self.model_provider == "HuggingFace":
            cfg_model = AutoConfig.from_pretrained(
                self.config["model_framework"]["HuggingFace"]["org_model"],
                num_labels=n_classes,
            )
            self.seg_model = AutoModelForSemanticSegmentation.from_pretrained(
                self.config["model_framework"]["HuggingFace"]["org_model"],
                config=cfg_model,
                ignore_mismatched_sizes=True,
            )

    def forward(self, x, met=None):
        output = None
        if self.model_provider == "SegmentationModelsPytorch":
            output = self.seg_model(x)
        elif self.model_provider == "HuggingFace":
            output = self.seg_model(x)
        return output


def get_module(checkpoint: str | Path) -> Mapping:
    if checkpoint is not None and os.path.isfile(checkpoint):
        weights = torch.load(checkpoint, map_location="cpu")
        if checkpoint.endswith(".ckpt"):  # type: ignore
            weights = weights["state_dict"]
    else:
        print(
            'Error with checkpoint provided: either a .ckpt with a "state_dict" key or an OrderedDict pt/pth file'
        )
        return {}

    if "model.seg_model" in list(weights.keys())[0]:
        weights = {k.partition("model.seg_model.")[2]: v for k, v in weights.items()}
        weights = {k: v for k, v in weights.items() if k != ""}

    return weights


#### LOADING MODEL ####
def load_model(config: dict) -> nn.Module:
    checkpoint = config["model_weights"]

    model_factory = FLAIR_ModelFactory(config)
    model = model_factory.seg_model

    state_dict = get_module(checkpoint=checkpoint)
    model.load_state_dict(state_dict=state_dict, strict=True)

    return model


def load_model_from_cfg_path(cfg_path: str) -> nn.Module:
    """
    Load a model from a given file.
    Args:
        cfg_path (str): Path to the configuration file.
    Returns:
        nn.Module: The loaded model.
    """

    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file {cfg_path} does not exist.")
    config = read_config({"conf": cfg_path})

    return load_model(config)


#### PREPARATION ####
def prepare_model(config: Config, device: torch.device) -> Config:
    # load one model, once only
    verbose = config.get("log_verbose", False)

    if verbose:
        print(
            f"""
    ##############################################
    ZONE DETECTION
    ##############################################
    """
        )

    arg_package = dict()

    weights_path = config.get("weights", "")
    onnx_flag = config.get("onnx", False) or weights_path.endswith(".onnx")

    model = load_model(config)

    if onnx_flag:
        if verbose:
            print(f"""    [ ] using ONNX model...""")
        model_type = "onnx"

        weights_path = get_onnx_path(model, config)
        onnx_path = onnx_optimize_model(config, weights_path)

        # create session
        ort_session = get_session(config, onnx_path)
        arg_package.update(
            {
                "ort_session": ort_session,
            }
        )

    else:
        model_type = "pytorch"
        dtype = getattr(torch, config.get("precision", "float31"), torch.float32)

        if verbose:
            print(
                f"""    [ ] using PyTorch model...

        CUDA available? {torch.cuda.is_available()}
        """
            )
        model = model.to(device)

        # optimization if necessary -> auxiliary function
        model = pt_optimize_model(config, model, verbose)

        arg_package.update(
            {
                "model": model,
                "device": device,
                "use_gpu": config["use_gpu"],
                "dtype": dtype,
            }
        )
    config.update({"model_type": model_type, "model_args": arg_package})

    print(f"""    [ ] warming up the model...""")
    warmup(model_type, config, arg_package)

    return config
