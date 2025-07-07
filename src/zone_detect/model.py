import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn

from pruna import SmashConfig, smash

import segmentation_models_pytorch as smp
from torchao.quantization import int8_weight_only, quantize_
from transformers import AutoModelForSemanticSegmentation, AutoConfig


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
        weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
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


def load_model(config: dict) -> nn.Module:
    checkpoint = config["model_weights"]

    model_factory = FLAIR_ModelFactory(config)
    model = model_factory.seg_model

    state_dict = get_module(checkpoint=checkpoint)

    precision = config.get("precision", "fp32")
    strict = precision == "fp32"  # allow mismatch if quantized

    model.load_state_dict(state_dict=state_dict, strict=strict)

    # check model data type
    # dtypes = set(param.dtype for param in model.parameters())
    # print("Model uses these data types:", dtypes)

    return model


def opti_pruna(model: nn.Module) -> nn.Module:
    """
    Apply pruna algorithms.
    """
    # config for example
    smash_config = SmashConfig()

    # smash_config["pruner"] = "torch_unstructured"
    # smash_config["torch_unstructured_pruning_method"] = "random"
    # smash_config["torch_unstructured_sparsity"] = 0.075
    smash_config["quantizer"] = "half"
    # smash_config["quantizer"] = "torch_dynamic"
    # smash_config["compiler"] = "torch_compile"

    model = smash(model=model, smash_config=smash_config)

    return model
