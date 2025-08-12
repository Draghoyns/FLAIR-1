# source:
# https://github.com/huggingface/optimum-quanto

import json
import os

import datetime
import torch
from optimum.quanto import (
    quantize,
    qint8,
    Calibration,
    freeze,
    requantize,
    quantization_map,
)
from safetensors.torch import save_file, load_file
from tqdm import tqdm

from src.zone_detect.optimization.calibration import load_calibration_images
from src.zone_detect.model import load_model_from_cfg_path


config_path = "/home/SHys/FLAIR-1/configs/20250804_config_detect_latest-update.yaml"
today = datetime.datetime.now().strftime("%Y%m%d")

model = load_model_from_cfg_path(config_path).eval().to("cuda")

quantize(model, weights=qint8, activations=qint8)


print(f"Model quantized successfully (fake quantization).")


# i understand samples are simple tensors, but what kind ? how many is advised ? where do i get them from ?
samples = torch.randn(1, 3, 512, 512).to("cuda")

samples_list = load_calibration_images("./0testing_saves/calibration_dataset")

with Calibration(momentum=0.9):
    with torch.no_grad():
        for batch in tqdm(samples_list, desc="Calibrating model with batches..."):
            model(batch)

            del batch
            torch.cuda.empty_cache()

freeze(model)

print(f"Model frozen successfully -> quantized to qint8.")

dir_path = f"./src/zone_detect/test/pytorch_various/{today}_quanto"

if not os.path.exists(dir_path):
    os.makedirs(dir_path)
print("Directory created successfully.")


save_file(
    model.state_dict(),
    f"./src/zone_detect/test/pytorch_various/{today}_quanto/model.safetensors",
)

print("Model state_dict saved successfully.")


with open(
    f"./src/zone_detect/test/pytorch_various/{today}_quanto/quantization_map.json", "w"
) as f:
    json.dump(quantization_map(model), f)


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
