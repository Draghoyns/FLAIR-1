from copy import deepcopy
import torch

import torch.nn as nn

from torchao.quantization.granularity import PerAxis, PerTensor
from torchao.quantization.observer import AffineQuantizedMinMaxObserver
from torchao.quantization.quant_primitives import MappingType
import tqdm
from transformers.models.upernet.modeling_upernet import UperNetConvModule

from src.zone_detect.model import load_model_from_cfg_path
from src.zone_detect.optimization.calibration import load_calibration_images
from src.zone_detect.utils import timer

config_path = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/configs/config_detect_compare_metrics.yaml"

model = load_model_from_cfg_path(config_path).eval().to("cuda")

'''
dtype = torch.float16
model = torch.compile(model, mode="reduce-overhead")

print("Model loaded and compiled successfully.")

toy_tensor = torch.randn(1, 3, 512, 512).to("cuda").to(dtype)

prediction = model(toy_tensor)
print("Prediction shape:", prediction.logits.shape)

print("Prediction done successfully.")

# per tensor input activation asymmetric quantization
act_obs = AffineQuantizedMinMaxObserver(
    MappingType.ASYMMETRIC,
    torch.uint8,
    granularity=PerTensor(),
    eps=torch.finfo(torch.float32).eps,
    scale_dtype=torch.float32,
    zero_point_dtype=torch.float32,
)

# per channel weight asymmetric quantization
weight_obs = AffineQuantizedMinMaxObserver(
    MappingType.ASYMMETRIC,
    torch.uint8,
    granularity=PerAxis(axis=0),
    eps=torch.finfo(torch.float32).eps,
    scale_dtype=torch.float32,
    zero_point_dtype=torch.float32,
)

leaf_modules = {
    name: module
    for name, module in model.named_modules()  # type: ignore
    if len(list(module.children())) == 0
}


total_layers = len(leaf_modules)
count = dict()

# module_types = set( [ nn.Conv2d, nn.Conv1d, nn.Linear, nn.ReLU, nn.BatchNorm2d, nn.LayerNorm, nn.Sequential, nn.Dropout, nn.AdaptiveAvgPool2d, nn.GELU, ])
module_types = set()

for name, module in leaf_modules.items():
    module_type = type(module)
    module_types.add(module_type)

for name, module in leaf_modules.items():
    for module_type in module_types:
        if isinstance(module, module_type):
            count[module_type] = count.get(module_type, 0) + 1
            break


for module_type, module_count in count.items():
    print(f"Found {module_count} {module_type.__name__} layers in the model")

print(f"Rests {total_layers - sum(count.values())} layers not in the list.")
print(f"Out of {total_layers} total layers.")

print(
    f"""Stats: 
Total layers: {total_layers}
Module types found: {len(module_types)}
Percentage for each type:
"""
)

for module_type, module_count in count.items():
    percentage = (module_count / total_layers * 100) if total_layers > 0 else 0
    print(f" - {module_type.__name__}: {percentage:.2f}%")

'''

model.qconfig = torch.ao.quantization.get_default_qconfig("x86")  # type: ignore

# Fuse the activations to preceding layers, where applicable.
# This needs to be done manually depending on the model architecture.
# Common fusions include `conv + relu` and `conv + batchnorm + relu`
# any way to do this a bit more adaptively to the model architecture ?
# add a model analysis metho to determine what to fuse, then do the fusion...?

# model_fp32_fused = torch.ao.quantization.fuse_modules( model, [["conv1", "relu1"], ["conv1", "bn1", "relu1"]], inplace=False)
model_fp32_fused = deepcopy(model)


def recursive_fuse(model):
    for name, module in model.named_children():
        # Try fusing if common patterns exist
        if isinstance(module, torch.nn.Sequential):
            try:
                torch.ao.quantization.fuse_modules(
                    module, [["0", "1"], ["0", "2", "3"]], inplace=True
                )
            except Exception:
                pass
        else:
            recursive_fuse(module)


recursive_fuse(model_fp32_fused)

# automatically inserts the observers into the model
model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)


# calibration
calibration_images = load_calibration_images(
    "/home/ign.fr/SHys/FLAIR-1/0testing_saves/20250730/calibration_dataset"
)
for image in calibration_images:
    image = image.unsqueeze(0).to("cuda")
    model_fp32_prepared(image)

print("Calibration done successfully.")


# then whatever quantization needed
model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
timing = {}
timed_inference = timer(timing)(model_int8)

res = timed_inference(calibration_images[0].unsqueeze(0).to("cuda"))

print("Quantized model prediction shape:", res.logits.shape)

print(f"Timing results: {timing}")
