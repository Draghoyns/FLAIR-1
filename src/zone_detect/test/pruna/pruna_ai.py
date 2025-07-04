# Testing pruna ai here :)

from pruna import SmashConfig, smash
import torch
from src.zone_detect.model import load_model

# define config
config = {
    "model_framework": {
        "model_provider": "SegmentationModelsPytorch",
        "HuggingFace": {"org_model": "openmmlab/upernet-swin-small"},
        "SegmentationModelsPytorch": {
            "encoder_decoder": "resnet34_unet",
        },
    },
    "model_weights": "/media/DATA/INFERENCE_HS/MODELS_IA/FLAIR1/unet_resnet/FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth",
    # "/media/DATA/INFERENCE_HS/MODELS_IA/FLAIR1/swin-upernet-small_IRV_SET1/checkpoints/ckpt-epoch=84-val_loss=0.37_00_HF_SwinUpernet_Small_IR-R-G_set1.ckpt",
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
model.to(device)

# SmashConfig

smash_config = SmashConfig()
quantization_option = "torch_dynamic"
# compiler_option = "torch_compile"

# smash_config["compiler"] = compiler_option
# smash_config["pruner"] = "torch_unstructured"
smash_config["quantizer"] = quantization_option
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
    logits = optimized_model(imgs)
    if config["model_framework"]["model_provider"] == "HuggingFace":
        logits = logits.logits

print("\tInference done, logits:")
print("\tShape:", logits.shape)

print("No error, we're good for step 2!\n")

print("Step 3 : saving optimized model...")
folder = "/home/ign.fr/SHys/FLAIR-1/0testing_saves"
optimized_model.save_pretrained(f"{folder}/20250630_pruna-{quantization_option}")
print("Model saved successfully!")

print("No error, we're good for step 3!\n")
print("Step 4 : loading optimized model...\n")
print("Loading the optimized model... (to be implemented)")
