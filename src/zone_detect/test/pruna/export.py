import torch
import json
from safetensors.torch import load_file
from pathlib import Path


def convert_safetensors_to_ckpt(model_folder_path, output_ckpt_path):
    """
    Convert a model folder with config.json and model.safetensors to a .ckpt file

    Args:
        model_folder_path: Path to folder containing config.json and model.safetensors
        output_ckpt_path: Path where to save the .ckpt file
    """
    model_folder = Path(model_folder_path)

    # Load the config
    config_path = model_folder / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load the model weights from safetensors
    safetensors_path = model_folder / "model.safetensors"
    state_dict = load_file(str(safetensors_path))

    # Load smash_config if it exists (Pruna-specific)
    smash_config = None
    smash_config_path = model_folder / "smash_config.json"
    if smash_config_path.exists():
        with open(smash_config_path, "r") as f:
            smash_config = json.load(f)

    # Create the checkpoint dictionary
    checkpoint = {
        "state_dict": state_dict,
        "config": config,
        "model_type": config.get("model_type", "upernet"),
        "torch_dtype": config.get("torch_dtype", "float32"),
        "transformers_version": config.get("transformers_version", "4.50.3"),
        "architectures": config.get(
            "architectures", ["UperNetForSemanticSegmentation"]
        ),
    }

    # Add smash_config if it exists (for Pruna compatibility)
    if smash_config:
        checkpoint["smash_config"] = smash_config

    # Add any other metadata that might be useful
    checkpoint["num_classes"] = len(config.get("id2label", {}))
    checkpoint["hidden_size"] = config.get("hidden_size", 512)

    # Save as checkpoint
    torch.save(checkpoint, output_ckpt_path)

    print(f"Successfully converted to {output_ckpt_path}")
    print(f"Model architecture: {config.get('architectures', ['Unknown'])[0]}")
    print(f"Number of classes: {len(config.get('id2label', {}))}")
    print(f"Torch dtype: {config.get('torch_dtype', 'Unknown')}")

    # Print some statistics
    total_params = sum(p.numel() for p in state_dict.values())
    total_size_mb = sum(p.numel() * p.element_size() for p in state_dict.values()) / (
        1024 * 1024
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_size_mb:.2f} MB")


def load_converted_ckpt(ckpt_path):
    """
    Load and inspect the converted checkpoint
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    print("Checkpoint keys:", list(checkpoint.keys()))
    print("State dict keys (first 10):", list(checkpoint["state_dict"].keys())[:10])

    # Check data types
    dtypes = {}
    for name, param in checkpoint["state_dict"].items():
        dtype_str = str(param.dtype)
        dtypes[dtype_str] = dtypes.get(dtype_str, 0) + 1

    print("Parameter data types:")
    for dtype, count in dtypes.items():
        print(f"  {dtype}: {count} tensors")

    return checkpoint


def safetensors_version(model_folder: str, output_ckpt: str) -> None:
    convert_safetensors_to_ckpt(model_folder, output_ckpt)

    # Verify the conversion
    print("\n" + "=" * 50)
    print("Verifying converted checkpoint:")
    load_converted_ckpt(output_ckpt)


import torch
import json
from pathlib import Path


def convert_pt_to_ckpt(
    model_pt_path, smash_config_path, output_ckpt_path, config_path=None
):
    """
    Convert a model.pt and smash_config.json to a .ckpt file

    Args:
        model_pt_path: Path to the model.pt file
        smash_config_path: Path to smash_config.json file
        output_ckpt_path: Path where to save the .ckpt file
        config_path: Optional path to config.json (if available)
    """

    # Load the model.pt file
    print(f"Loading model from: {model_pt_path}")
    model_data = torch.load(model_pt_path, map_location="cpu", weights_only=False)

    # Load the smash_config
    print(f"Loading smash config from: {smash_config_path}")
    with open(smash_config_path, "r") as f:
        smash_config = json.load(f)

    # Load regular config if available
    config = None
    if config_path and Path(config_path).exists():
        print(f"Loading config from: {config_path}")
        with open(config_path, "r") as f:
            config = json.load(f)

    # Analyze the structure of model.pt
    print("\nAnalyzing model.pt structure...")
    print(f"Type of loaded data: {type(model_data)}")

    if isinstance(model_data, dict):
        print("Keys in model.pt:", list(model_data.keys()))

        # Extract state_dict - it might be stored under different keys
        state_dict = None
        if "state_dict" in model_data:
            state_dict = model_data["state_dict"]
        elif "model" in model_data:
            state_dict = model_data["model"]
        elif "model_state_dict" in model_data:
            state_dict = model_data["model_state_dict"]
        else:
            # If no clear state_dict key, assume the whole thing is the state_dict
            # but filter out non-tensor values
            state_dict = {
                k: v for k, v in model_data.items() if isinstance(v, torch.Tensor)
            }
            if not state_dict:
                # If no tensors found, use the whole dict
                state_dict = model_data
    else:
        # If model_data is not a dict, it might be a model object or state_dict directly
        if hasattr(model_data, "state_dict"):
            state_dict = model_data.state_dict()
        else:
            state_dict = model_data

    # Create the checkpoint dictionary
    checkpoint = {
        "state_dict": state_dict,
        "smash_config": smash_config,
    }

    # Add original model metadata if it exists
    if isinstance(model_data, dict):
        # Preserve non-tensor metadata from original model.pt
        for key, value in model_data.items():
            if key not in [
                "state_dict",
                "model",
                "model_state_dict",
            ] and not isinstance(value, torch.Tensor):
                checkpoint[key] = value

    # Add config if provided
    if config:
        checkpoint["config"] = config
        # Add useful metadata from config
        if "model_type" in config:
            checkpoint["model_type"] = config["model_type"]
        if "architectures" in config:
            checkpoint["architectures"] = config["architectures"]
        if "torch_dtype" in config:
            checkpoint["torch_dtype"] = config["torch_dtype"]
        if "id2label" in config:
            checkpoint["num_classes"] = len(config["id2label"])

    # Add metadata from smash_config if available
    if "compression_info" in smash_config:
        checkpoint["compression_info"] = smash_config["compression_info"]

    # Save as checkpoint
    torch.save(checkpoint, output_ckpt_path)

    print(f"\nSuccessfully converted to {output_ckpt_path}")

    # Print statistics
    if isinstance(state_dict, dict):
        total_params = sum(
            p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor)
        )
        total_size_mb = sum(
            p.numel() * p.element_size()
            for p in state_dict.values()
            if isinstance(p, torch.Tensor)
        ) / (1024 * 1024)
        print(f"Total parameters: {total_params:,}")
        print(f"Model size: {total_size_mb:.2f} MB")

        # Check data types
        dtypes = {}
        for name, param in state_dict.items():
            if isinstance(param, torch.Tensor):
                dtype_str = str(param.dtype)
                dtypes[dtype_str] = dtypes.get(dtype_str, 0) + 1

        print("Parameter data types:")
        for dtype, count in dtypes.items():
            print(f"  {dtype}: {count} tensors")

    print(f"Smash config keys: {list(smash_config.keys())}")


def inspect_pt_file(model_pt_path):
    """
    Inspect the structure of a .pt file to understand its contents
    """
    print(f"Inspecting: {model_pt_path}")
    model_data = torch.load(model_pt_path, map_location="cpu", weights_only=False)

    print(f"Type: {type(model_data)}")

    if isinstance(model_data, dict):
        print("Dictionary keys:")
        for key, value in model_data.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: torch.Tensor {value.shape} {value.dtype}")
            elif isinstance(value, dict):
                print(f"  {key}: dict with {len(value)} items")
                # Show first few keys if it's a state_dict-like structure
                if len(value) > 0:
                    sample_keys = list(value.keys())[:3]
                    print(f"    Sample keys: {sample_keys}")
            else:
                print(f"  {key}: {type(value)} - {str(value)[:100]}")
    elif hasattr(model_data, "state_dict"):
        print("Model object detected")
        state_dict = model_data.state_dict()
        print(f"State dict has {len(state_dict)} parameters")
        sample_keys = list(state_dict.keys())[:5]
        print(f"Sample parameter names: {sample_keys}")
    else:
        print(f"Unknown format: {type(model_data)}")


def load_converted_ckpt_v2(ckpt_path):
    """
    Load and inspect the converted checkpoint
    """
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    print("Checkpoint keys:", list(checkpoint.keys()))

    if "state_dict" in checkpoint:
        print("State dict keys (first 10):", list(checkpoint["state_dict"].keys())[:10])

    if "smash_config" in checkpoint:
        print("Smash config keys:", list(checkpoint["smash_config"].keys()))

    return checkpoint


def pt_version(folder_path: str, output_ckpt: str) -> None:

    model_folder = Path(folder_path)

    # Load the config
    smash_config_path = model_folder / "smash_config.json"

    # Load the model weights from safetensors
    model_path = model_folder / "optimized_model.pt"

    # Load smash_config if it exists (Pruna-specific)
    config_path = model_folder / "config.json"
    if not config_path.exists():
        config_path = None

    # First, inspect your .pt file to understand its structure
    print("=" * 60)
    print("INSPECTING MODEL.PT FILE")
    print("=" * 60)
    inspect_pt_file(str(model_path))

    print("\n" + "=" * 60)
    print("CONVERTING TO CKPT")
    print("=" * 60)

    # Convert pt + smash_config to ckpt
    convert_pt_to_ckpt(
        model_pt_path=str(model_path),
        smash_config_path=str(smash_config_path),
        output_ckpt_path=str(model_folder / output_ckpt),
        config_path=str(config_path) if config_path else None,
    )

    print("\n" + "=" * 60)
    print("VERIFYING CONVERTED CHECKPOINT")
    print("=" * 60)
    load_converted_ckpt(f"{model_folder}/converted_model.ckpt")


# Example usage
if __name__ == "__main__":
    # Convert safetensors + config to ckpt
    model_folder = (
        "/home/ign.fr/SHys/FLAIR-1/0testing_saves/20250630_pruna-torch_dynamic"
    )
    output_ckpt = f"{model_folder}/converted_model.ckpt"

    pt_version(
        folder_path=model_folder,
        output_ckpt=output_ckpt,
    )

    # safetensors_version( model_folder=model_folder, output_ckpt=output_ckpt,)

    original_ckpt = "/media/DATA/INFERENCE_HS/MODELS_IA/FLAIR1/swin-upernet-small_IRV_SET1/checkpoints/ckpt-epoch=84-val_loss=0.37_00_HF_SwinUpernet_Small_IR-R-G_set1.ckpt"

    load_converted_ckpt(output_ckpt)
