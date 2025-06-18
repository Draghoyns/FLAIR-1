# Testing pruna ai here :)

from pruna import SmashConfig, smash
import torch
from src.zone_detect.model import load_model

# define config
config = {
    "model_framework": {
        "model_provider": "HuggingFace",
        "HuggingFace": {"org_model": "openmmlab/upernet-swin-small"},
    },
    "model_weights": "/media/DATA/INFERENCE_HS/MODELS_IA/FLAIR1/swin-upernet-small_IRV_SET1/checkpoints/ckpt-epoch=84-val_loss=0.37_00_HF_SwinUpernet_Small_IR-R-G_set1.ckpt",
    "channels": [
        1,
        2,
        3,
    ],
    "n_classes": 19,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load the model

model = load_model(config)

# SmashConfig

smash_config = SmashConfig()
# smash_config["compiler"] = "torch_compile"
smash_config["pruner"] = "torch_unstructured"
smash_config["quantizer"] = "half"
# Smash !!

optimized_model = smash(model=model, smash_config=smash_config)
# returns a PrunaModel object

print("Trying things out with Pruna AI...")

optimized_model.eval()
optimized_model.to(device)

print("No error, we're good for step 1!\n")

print("Step 2 : inference...")

imgs = torch.randn(2, 3, 512, 512).to(device)

with torch.no_grad():
    logits = optimized_model(imgs).logits

print("\tInference done, logits:")
print("\tShape:", logits.shape)

print("No error, we're good for step 2!\n")
