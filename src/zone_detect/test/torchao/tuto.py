import os
import torch
from torch import nn
import copy
from torchao.quantization import int8_weight_only, quantize_, Int8WeightOnlyConfig
import torchao

from src.zone_detect.model import FLAIR_ModelFactory, get_module, load_model

print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Compute capability: {torch.cuda.get_device_capability()}")
print(f"CUDA version: {torch.version.cuda}")  # type: ignore
print(f"PyTorch version: {torch.__version__}")
print(f"BFloat16 support: {torch.cuda.is_bf16_supported()}")

# Check if your GPU supports the required compute capability
# TorchAO INT4 typically requires compute capability 8.0+
major, minor = torch.cuda.get_device_capability()
if major < 8:
    print(
        f"WARNING: Your GPU compute capability {major}.{minor} may not support TorchAO INT4 quantization"
    )

print(f"PyTorch version: {torch.__version__}")
print(f"TorchAO version: {torchao.__version__}")

# load model
ckpt_ru = "/media/DATA/INFERENCE_HS/MODELS_IA/FLAIR1/unet_resnet/FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth"
ckpt_swin = "/media/DATA/INFERENCE_HS/MODELS_IA/FLAIR1/swin-upernet-small_IRV_SET1/checkpoints/ckpt-epoch=84-val_loss=0.37_00_HF_SwinUpernet_Small_IR-R-G_set1.ckpt"

model_config = {
    "model_framework": {
        "model_provider": "HuggingFace",
        "HuggingFace": {"org_model": "openmmlab/upernet-swin-small"},
        "SegmentationModelsPytorch": {
            "encoder_decoder": "resnet34_unet",
        },
    },
    "model_weights": ckpt_swin,
    "channels": [
        1,
        2,
        3,
    ],
    "n_classes": 19,
}

model = load_model(model_config)

print("Model loaded successfully!")


model_config.update({"precision": "int8"})  # or "int8_full"

# model = model.eval().to("cuda")
model = model.eval()  # from torchao tutorial

# try compiling (optional) (from torchao tutorial)
# TODO: further check docs
# model = torch.compile(model, mode="max-autotune", fullgraph=True)

model_original = copy.deepcopy(model)  # keep original
model = model.to(torch.bfloat16)

precision = model_config.get("precision", "fp32")

# quantization
if precision.startswith("int8"):
    # quantize_(model, Int8WeightOnlyConfig())  # type: ignore
    quantize_(model, int8_weight_only(group_size=32))  # type: ignore
    name, tensor = list(model.state_dict().items())[0]
    print(
        f"""Model quantized to {model.dtype} successfully!
Model type after quantization: {type(model)}
    """
    )


# saving
save_dir = "/home/ign.fr/SHys/FLAIR-1/0testing_saves"
if precision.endswith("full"):
    torch.save(model, f"{save_dir}/model_{precision}.pth")
else:
    torch.save(model.state_dict(), f"{save_dir}/model_{precision}.pth")
torch.save(model_original.state_dict(), f"{save_dir}/model_original.pth")

print("Model quantized and saved successfully!\n")

# compare sizes
int4_size_mb = os.path.getsize(f"{save_dir}/model_{precision}.pth") / (1024 * 1024)
original_size_mb = os.path.getsize(f"{save_dir}/model_original.pth") / (1024 * 1024)

print(f"{precision} model size: {int4_size_mb:.2f} MB")
print(f"Original model size: {original_size_mb:.2f} MB")

print("Yay everything went well! (for now...)")

# tuto ends here
# ___________________________________________________#
# now we load the model to run inference

model_path = f"{save_dir}/model_{precision}.pth"
model_config["model_weights"] = model_path


def custom_load_model(config: dict) -> nn.Module:
    checkpoint = config["model_weights"]

    model_factory = FLAIR_ModelFactory(config)
    model = model_factory.seg_model

    # state_dict = get_module(checkpoint=checkpoint)

    state_dict = torch.load(checkpoint, map_location="cpu")

    print(f"State_dict: {state_dict.items()}")

    precision = config.get("precision", "fp32")
    strict = precision == "fp32"  # allow mismatch if quantized

    print(
        f"""Model skeleton is in {model.dtype} successfully!
Model type after quantization: {type(model)}
    """
    )
    # print(f"Model: {model.state_dict().items()}")

    model.load_state_dict(
        state_dict=state_dict,
        strict=strict,
    )
    """raise NotImplementedError(
        "This is a test file for TorchAO, not for inference. Please use the dedicated falir-detect command instead"
    )
    if precision == "int8":

        model = model.eval().to(torch.bfloat16)
        quantize_(model, Int8WeightOnlyConfig(group_size=32))  # type: ignore
        model = model.to("cuda")

        for name, module in model.named_modules():
            if "QuantLinear" in str(type(module)):
                print(f"{name}: Quantized Linear layer")
            else:
                print(f"{name}: {type(module).__name__}")"""

    return model


model = custom_load_model(model_config)

print("Model loaded for inference!")
print(f"Model type: {type(model)}")
