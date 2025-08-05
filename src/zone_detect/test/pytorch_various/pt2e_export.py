import torch

from src.zone_detect.model import load_model_from_cfg_path

config_path = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/configs/config_detect_compare_metrics.yaml"

model = load_model_from_cfg_path(config_path)

example_inputs = (torch.randn(1, 3, 512, 512),)
model = model.eval()

# Step 1. program capture
# This is available for pytorch 2.6+, for more details on lower pytorch versions
# please check `Export the model with torch.export` section
m = torch.export.export(model, example_inputs).module()
# we get a model with aten ops

print("Model exported successfully.")
