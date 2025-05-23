import datetime
import os
from pathlib import Path
import numpy as np
import rasterio
import torch
import yaml

from src.zone_detect.test.tiles import get_stride


#### CONFIG ####
def read_config(args) -> dict:
    file_path = args.conf
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    # put arguments in config
    config["metrics"] = args.metrics
    config["batch_mode"] = args.batch_mode
    config["compare"] = args.compare

    return preprocess_config(config)


def preprocess_config(config: dict) -> dict:
    """Clean the config file by formatting correctly
    and raising obvious errors before any run."""

    # paths
    # check existence
    Path(config["output_path"]).mkdir(parents=True, exist_ok=True)
    assert os.path.exists(config["input_img_path"]), "Input image path does not exist."
    config["input_img_path"] = Path(config["input_img_path"]).with_suffix(".tif")

    if config["metrics"]:
        config["metrics_out"] = config["output_path"] + "/metrics.json"
        assert os.path.exists(config["truth_path"]), "Ground truth path does not exist."

        config["truth_path"] = Path(config["truth_path"]).with_suffix(".tif")

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
        "class_prob",
        "argmax",
    ], "Invalid output type: should be argmax or class_prob."
    assert type(config["n_classes"]) == int, "n_classes should be an integer"
    assert config["norma_task"][0]["norm_type"] in [
        "custom",
        "scaling",
    ], "Invalid normalization type: should be custom or scaling."

    # model
    assert os.path.isfile(config["model_weights"]), "Model weights file does not exist."
    if os.path.splitext(config["model_weights"])[1] not in [".pth", ".ckpt"]:
        raise ValueError(
            "Model weights should be a .pth or .ckpt file. "
            f"Got {os.path.splitext(config['model_weights'])[1]}"
        )

    if config["compare"]:

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


def gen_param_combination(config: dict) -> list:
    """Generate all possible combinations of parameters.
    Handles single case or iterative case."""
    combi = []

    # TODO : add differet padding strategies
    padding_list = config.get("strategies", {}).get("padding_overall", [])
    if not padding_list:
        padding_list = ["no-padding"]

    # configuration for comparison
    tiling_cfg = config.get("strategies", {}).get("tiling", {})
    if tiling_cfg.get("enabled", False):
        tile_size_list = tiling_cfg.get("size_range", [config["img_pixels_detection"]])
    else:
        tile_size_list = [config["img_pixels_detection"]]

    stitching_cfg = config.get("strategies", {}).get("stitching", {})
    # default stitching : exact clipping
    if stitching_cfg.get("enabled", False):
        margin_list = stitching_cfg.get("margin", [config["margin"]])
        stitching_methods = stitching_cfg.get("methods", ["exact-clipping"])
    else:
        margin_list = [config["margin"]]
        stitching_methods = ["exact-clipping"]

    for padding in padding_list:
        for img_pixels_detection in tile_size_list:
            for margin in margin_list:
                if margin < 1:
                    margin = int(margin * img_pixels_detection)
                # skip if parameters are not valid
                if img_pixels_detection <= 2 * margin:
                    print(
                        f"""    [x] skipping {img_pixels_detection} pixels detection size with {margin} margin..."""
                    )
                    continue

                # avoid mutation
                tmp_config = config.copy()
                tmp_config["margin"] = margin
                tmp_config["img_pixels_detection"] = img_pixels_detection

                stride_list = get_stride(tmp_config)

                for stride in stride_list:
                    for stitch in stitching_methods:

                        param = {
                            "img_pixels_detection": img_pixels_detection,
                            "margin": margin,
                            "padding": padding,
                            "stitching": stitch,
                            "stride": stride,
                        }
                        combi.append(param)

    return combi


def extract_method(method: str, info: dict = {}) -> dict:
    """Extract the method parameters from the method name."""
    elements = method.split("_")
    for param in elements:
        if param.startswith("size="):
            info["patch_size"] = int(param.split("=")[1])
        elif param.startswith("stride="):
            info["stride"] = int(param.split("=")[1])
        elif param.startswith("margin="):
            info["margin"] = int(param.split("=")[1])
        elif param.startswith("padding="):
            info["padding"] = param.split("=")[1]
        elif param.startswith("stitching="):
            info["stitching"] = param.split("=")[1]
        else:
            param = param.split("=")
            info[param[0]] = param[1]

    return info


def info_extract(file: Path) -> dict:
    """Extract the information from the filename, namely the region and the method used.
    Args:
        filename (Path): the filename to extract the information from. Should be full path.
    Returns:
        dict: with the keys [dpt, zone, patch_size, stride, margin, padding, stitching] and other if they exist.
    """
    filename = str(file)

    if not filename.endswith(".tif"):
        raise ValueError("Filename should end with .tif what are you doing ?")
    name = filename.split("/")[-1]
    name = name.split(".")[0]
    info = {}
    region_type, method = name.split("-ARGMAX-S_")
    # region info
    region_type = region_type.split("_")
    dpt, zone, data_type = region_type[:2], region_type[2:-1], region_type[-1]
    if not dpt[0].startswith("D"):
        info["dpt"] = "D" + "_".join(dpt)
    info["zone"] = "_".join(zone)

    # method info
    info["method"] = method
    info = extract_method(method, info)

    return info


#### SETUP ####
def setup_out_path(config: dict) -> dict:
    """Setup the output directory"""
    output = Path(config["output_path"])
    output.mkdir(parents=True, exist_ok=True)
    child_dir = output

    if config["compare"]:
        # create a directory with a unique id
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        child_dir = child_dir / Path(current_time)
        os.makedirs(child_dir, exist_ok=True)
        print(f"Creating output directory: {child_dir}")

    config["local_out"] = child_dir

    return config


def setup_device(config: dict) -> tuple[torch.device, bool]:
    """Setup the device"""

    use_gpu = False if torch.cuda.is_available() is False else config["use_gpu"]
    device = torch.device("cuda" if use_gpu else "cpu")

    return device, use_gpu


def setup(args) -> tuple[dict, torch.device, bool]:
    """Setup the device"""
    config = read_config(args)
    device, use_gpu = setup_device(config)

    return config, device, use_gpu


def setup_indiv_path(config: dict, identifier: str) -> tuple[dict, str]:
    """Setup the output path for individual images"""

    out_name = config["output_name"] + identifier

    if not out_name.endswith(".tif"):
        out_name += ".tif"

    try:
        base_name = out_name
        path_out = os.path.join(config["local_out"], base_name)

        # Do not overwrite if file exists, add counter
        filename, ext = os.path.splitext(base_name)
        counter = 1

        while os.path.exists(path_out):
            new_name = f"{filename}_{counter}{ext}"
            path_out = os.path.join(config["local_out"], new_name)
            counter += 1
        return config, path_out
    except Exception as error:
        print(f"Something went wrong during detection configuration: {error}")
        raise error  # avoid silent failure


def open_images(config: dict, local_out: Path, get_truth: bool):
    """Get the input image array and the ground truth if necessary."""

    if get_truth:
        full_truth_path = Path(config["truth_path"])
        with rasterio.open(full_truth_path) as src:
            truth_array = src.read(1) - 1  # to start from 0

        # common to a zone (= one tif image)
        dpt, zone = Path(config["input_img_path"]).parts[-3:-1]
        metrics_json = local_out / Path(f"metrics_per-patch_{dpt}_{zone}.json")
    else:
        truth_array = np.zeros((1, 1), dtype=np.uint8)
        metrics_json = Path()

    return truth_array, metrics_json


#### ROUNDING AND ALIGNING ####
def truncate(value: float, decimals: int) -> float:
    """Truncate a float to a given number of decimal places"""
    factor = 10**decimals
    rounded = round(value, 2) * factor
    return rounded / factor
