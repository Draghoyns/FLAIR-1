import datetime
import os
from pathlib import Path
import torch
import yaml


#### CONFIG ####
def read_config(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


#### SETUP ####
def setup_out_path(config: dict, args) -> tuple[dict, bool]:
    """Setup the output directory"""
    Path(config["output_path"]).mkdir(parents=True, exist_ok=True)

    if args.compare:
        # create a directory with a unique id
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        child_dir = os.path.join(config["output_path"], current_time)
        os.makedirs(child_dir, exist_ok=True)
        config["output_path"] = child_dir

    return config, args.compare


def setup_device(config: dict) -> tuple[torch.device, bool]:
    """Setup the device"""

    use_gpu = False if torch.cuda.is_available() is False else config["use_gpu"]
    device = torch.device("cuda" if use_gpu else "cpu")

    return device, use_gpu


def setup(args) -> tuple[dict, torch.device, bool, bool]:
    """Setup the device and output path"""
    config = read_config(args.conf)
    device, use_gpu = setup_device(config)
    config, compare = setup_out_path(config, args)

    return config, device, use_gpu, compare


def setup_indiv_path(config: dict, identifier: str = "") -> tuple[dict, str]:
    """Setup the output path for individual images"""
    assert isinstance(config["output_path"], str), "Output path does not exist."
    assert os.path.exists(config["input_img_path"]), "Input image path does not exist."
    assert (
        config["margin"] * 2 < config["img_pixels_detection"]
    ), "Margin is too large : margin*2 < img_pixels_detection"
    assert config["output_type"] in [
        "class_prob",
        "argmax",
    ], "Invalid output type: should be argmax or class_prob."
    assert config["norma_task"][0]["norm_type"] in [
        "custom",
        "scaling",
    ], "Invalid normalization type: should be custom or scaling."
    assert os.path.isfile(config["model_weights"]), "Model weights file does not exist."

    out_name = config["output_name"] + identifier

    if not out_name.endswith(".tif"):
        out_name += ".tif"

    try:
        # Path(config['output_path']).mkdir(parents=True, exist_ok=True)
        base_name = out_name
        path_out = os.path.join(config["output_path"], base_name)

        # Do not overwrite if file exists, add counter
        filename, ext = os.path.splitext(base_name)
        counter = 1

        while os.path.exists(path_out):
            new_name = f"{filename}_{counter}{ext}"
            path_out = os.path.join(config["output_path"], new_name)
            counter += 1
        # config['output_name'] = os.path.splitext(os.path.basename(path_out))[0]
        return config, path_out
    except Exception as error:
        print(f"Something went wrong during detection configuration: {error}")
        raise error  # avoid silent failure


#### PATH HANDLING ####
def valid_truth(config: dict) -> Path:
    """Check if the ground truth path is valid and coherent with the input path :
    the zone should be the same in both paths.
    """
    truth_path = config["truth_path"]
    # verify coherence with input path
    sanity_check = config["input_img_path"].split("/")[-3:-1]  # zone
    if truth_path.split("/")[-3:-1] != sanity_check:
        raise ValueError(
            f"Ground truth path {truth_path} does not match input path {config['input_img_path']}"
        )
    return Path(truth_path)


#### ROUNDING AND ALIGNING ####
def truncate(value: float, decimals: int) -> float:
    """Truncate a float to a given number of decimal places"""
    factor = 10**decimals
    rounded = round(value, 2) * factor
    return rounded / factor
