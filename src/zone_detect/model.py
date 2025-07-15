import os

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn

from pruna import SmashConfig, smash

import segmentation_models_pytorch as smp
from transformers import AutoModelForSemanticSegmentation, AutoConfig

from src.zone_detect.utils import read_config

from src.zone_detect.utils import read_config


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


def opti_pruna(model: nn.Module, sparse: float = 0) -> nn.Module:
    """
    Apply pruna algorithms.
    """

    print(f"Applying pruna with sparsity: {sparse:.2%}")

    # config for example
    smash_config = SmashConfig()

    smash_config["pruner"] = "torch_unstructured"
    # smash_config["torch_unstructured_pruning_method"] = "random"
    smash_config["torch_unstructured_sparsity"] = sparse
    # smash_config["quantizer"] = "half"
    # smash_config["quantizer"] = "torch_dynamic"
    # smash_config["compiler"] = "torch_compile"

    model = smash(model=model, smash_config=smash_config)

    return model


def analyze_model_weights(
    model: nn.Module,
    include_bias: bool = True,
    epsilon: float = 1e-3,
    save: bool = False,
) -> list:
    report = []
    group_stats = {}
    total_near_zeros = 0
    total_elements = 0

    for name, param in model.named_parameters():
        if not include_bias and "bias" in name:
            continue
        if param.requires_grad:
            data = param.data.cpu().numpy()
            abs_data = np.abs(data)
            max_val = abs_data.max()

            if max_val == 0:
                near_zero_mask = abs_data == 0
            else:
                threshold = epsilon * max_val
                near_zero_mask = abs_data < threshold

            near_zero_count = near_zero_mask.sum()
            element_count = data.size

            total_near_zeros += near_zero_count
            total_elements += element_count

            layer_type = name.split(".")[0] if "." in name else name
            if layer_type not in group_stats:
                group_stats[layer_type] = {"near_zeros": 0, "elements": 0}
            group_stats[layer_type]["near_zeros"] += near_zero_count
            group_stats[layer_type]["elements"] += element_count

            if save:
                report.append(
                    {
                        "Layer.Parameter": name,
                        "Shape": list(data.shape),
                        "Mean": data.mean(),
                        "Std": data.std(),
                        "Min": data.min(),
                        "Max": data.max(),
                        "Near-Zeros (<{:.0e} * max)".format(epsilon): near_zero_count,
                        "Near-Zero Ratio": near_zero_count / element_count,
                    }
                )

    overall_sparsity = total_near_zeros / total_elements if total_elements > 0 else 0
    print(
        f"\n🔍 Overall near-zero sparsity (< {epsilon:.0e} * max): {overall_sparsity:.4%}"
    )

    print("\n📊 Sparsity by Principal Layer:")
    for group_name, stats in group_stats.items():
        group_sparsity = (
            stats["near_zeros"] / stats["elements"] if stats["elements"] > 0 else 0
        )
        print(f"{group_name}: {group_sparsity:.4%}")

    return report
