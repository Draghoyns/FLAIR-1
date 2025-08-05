# source:
# https://github.com/huggingface/optimum-quanto

import datetime
import torch
from src.zone_detect.model import load_model_from_cfg_path
from optimum.quanto import quantize, qint8

config_path = "/media/stores/tmp/store-DAI/pocs/INFERENCE_SH/DATA/inference_flair/configs/config_detect_compare_metrics.yaml"
today = datetime.datetime.now().strftime("%Y%m%d")

model = load_model_from_cfg_path(config_path).eval().to("cuda")

quantize(model, weights=qint8, activations=qint8)

print(f"Model quantized successfully to {model.dtype}.")

from optimum.quanto import Calibration

# i understand samples are simple tensors, but what kind ? how many is advised ? where do i get them from ?
samples = torch.randn(1, 3, 512, 512).to("cuda")

# with Calibration(momentum=0.9):
#   model(samples)

from optimum.quanto import freeze

freeze(model)

dir_path = f"./src/zone_detect/test/pytorch_various/{today}_quanto"
import os

if not os.path.exists(dir_path):
    os.makedirs(dir_path)
print("Directory created successfully.")


from safetensors.torch import save_file

save_file(
    model.state_dict(),
    f"./src/zone_detect/test/pytorch_various/{today}_quanto/model.safetensors",
)

print("Model state_dict saved successfully.")

import json

from optimum.quanto import quantization_map

with open(
    f"./src/zone_detect/test/pytorch_various/{today}_quanto/quantization_map.json", "w"
) as f:
    json.dump(quantization_map(model), f)


from safetensors.torch import load_file
from optimum.quanto import requantize


state_dict = load_file(
    f"./src/zone_detect/test/pytorch_various/{today}_quanto/model.safetensors"
)
with open(
    f"./src/zone_detect/test/pytorch_various/{today}_quanto/quantization_map.json", "r"
) as f:
    quantization_map = json.load(f)

# Create an empty model from your modeling code and requantize it
new_model = load_model_from_cfg_path(config_path)
requantize(new_model, state_dict, quantization_map, device=torch.device("cuda"))

print("Model requantized successfully.")

# let's try inference

input = torch.randn(1, 3, 512, 512).to("cuda")
output = new_model(input)
print(f"Output shape: {output.logits.shape}")
print("Inference done successfully.")
