import datetime
import os
from pathlib import Path
import torch
import yaml


#### CONFIG ####
def read_config(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def preprocess_config(config: dict, compare: bool) -> dict:
    """Clean the config file by formatting correctly
    and raising obvious errors before any run."""

    # paths
    # check existence ?
    config["input_img_path"] = Path(config["input_img_path"]).with_suffix(".tif")
    config["truth_path"] = Path(config["truth_path"]).with_suffix(".tif")
    config["metrics_out"] = Path(config["metrics_out"]).with_suffix(".json")

    # channels
    assert isinstance(config["channels"], list) and all(
        isinstance(c, int) for c in config["channels"]
    ), "Channels should be a list of integers"

    # inference param
    assert (
        type(config["img_pixels_detection"]) == int
    ), "img_pixels_detection should be an integer"
    assert (
        type(config["margin"]) == int
        and 2 * config["margin"] < config["img_pixels_detection"]
    ), "Margin should be an integer and less than half of img_pixels_detection"
    assert config["output_type"] in [
        "argmax",
        "class_prob",
    ], "Output type should be either argmax or class_prob"
    assert type(config["n_classes"]) == int, "n_classes should be an integer"

    # model
    if os.path.splitext(config["model_weights"])[1] not in [".pth", ".ckpt"]:
        raise ValueError(
            "Model weights should be a .pth or .ckpt file. "
            f"Got {os.path.splitext(config['model_weights'])[1]}"
        )

    if compare:

        config["strategies"]["tiling"]["size_range"] = check_list_type(
            config["strategies"]["tiling"]["size_range"], int
        )
        config["strategies"]["tiling"]["stride_range"] = check_list_type(
            config["strategies"]["tiling"]["stride_range"], float
        )
        assert all(
            i >= 0 and i <= 1 for i in config["strategies"]["tiling"]["stride_range"]
        ), "Stride should be a percentage"
        config["strategies"]["stitching"]["methods"] = check_list_type(
            config["strategies"]["stitching"]["methods"], str
        )
        config["strategies"]["stitching"]["margin"] = check_list_type(
            config["strategies"]["stitching"]["margin"], float
        )
        assert all(
            i >= 0 and i <= 1 for i in config["strategies"]["stitching"]["margin"]
        ), "Margin should be a percentage"

    return config


def check_list_type(lst: list, expected_type: type) -> list:
    res = lst
    if isinstance(lst, expected_type):
        res = [lst]
    elif hasattr(lst, "__iter__"):
        res = [i for i in lst if isinstance(i, expected_type)]

    assert all(
        isinstance(i, expected_type) for i in res
    ), f"List should be of type {expected_type}"
    return res


#### SETUP ####
def setup_out_path(config: dict, compare: bool) -> tuple[dict, bool]:
    """Setup the output directory"""
    Path(config["output_path"]).mkdir(parents=True, exist_ok=True)

    if compare:
        # create a directory with a unique id
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        child_dir = os.path.join(config["output_path"], current_time)
        os.makedirs(child_dir, exist_ok=True)
        config["local_out"] = child_dir

    return config, compare


def setup_device(config: dict) -> tuple[torch.device, bool]:
    """Setup the device"""

    use_gpu = False if torch.cuda.is_available() is False else config["use_gpu"]
    device = torch.device("cuda" if use_gpu else "cpu")

    return device, use_gpu


def setup(args) -> tuple[dict, torch.device, bool, bool]:
    """Setup the device and output path"""
    config = read_config(args.conf)
    device, use_gpu = setup_device(config)
    config, compare = setup_out_path(config, args.compare)

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
        path_out = os.path.join(config["local_out"], base_name)

        # Do not overwrite if file exists, add counter
        filename, ext = os.path.splitext(base_name)
        counter = 1

        while os.path.exists(path_out):
            new_name = f"{filename}_{counter}{ext}"
            path_out = os.path.join(config["local_out"], new_name)
            counter += 1
        # config['output_name'] = os.path.splitext(os.path.basename(path_out))[0]
        return config, path_out
    except Exception as error:
        print(f"Something went wrong during detection configuration: {error}")
        raise error  # avoid silent failure


#### ROUNDING AND ALIGNING ####
def truncate(value: float, decimals: int) -> float:
    """Truncate a float to a given number of decimal places"""
    factor = 10**decimals
    rounded = round(value, 2) * factor
    return rounded / factor
