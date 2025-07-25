from pathlib import Path
from geopandas import GeoDataFrame
import rasterio
import torch

from torch.utils.data import DataLoader

from typing import Any

from src.zone_detect.dataset import Sliced_Dataset
from src.zone_detect.inference import warmup
from src.zone_detect.model import load_model

from src.zone_detect.optimization.onnx import get_session, onnx_optimize_model
from src.zone_detect.optimization.pytorch import pt_optimize_model
from src.zone_detect.slicing_job import slice_extent
from src.zone_detect.test.onnx.onnx_export import get_onnx_path
from src.zone_detect.utils import conf_log, setup_indiv_path

Config = dict[str, Any]


def prepare_model(config: Config, device: torch.device) -> Config:
    # load one model, once only
    verbose = config.get("log_verbose", False)

    if verbose:
        print(
            f"""
    ##############################################
    ZONE DETECTION
    ##############################################
    """
        )

    arg_package = dict()

    weights_path = config.get("weights", "")
    onnx_flag = config.get("onnx", False) or weights_path.endswith(".onnx")

    if onnx_flag:
        if verbose:
            print(f"""    [ ] using ONNX model...""")
        model_type = "onnx"

        weights_path = get_onnx_path(config)
        onnx_path = onnx_optimize_model(config, weights_path)

        # create session
        ort_session = get_session(config, onnx_path)
        arg_package.update(
            {
                "ort_session": ort_session,
            }
        )

    else:
        model_type = "pytorch"
        model = load_model(config)
        dtype = getattr(torch, config.get("precision", "float32"), torch.float32)

        if verbose:
            print(
                f"""    [ ] using PyTorch model...

        CUDA available? {torch.cuda.is_available()}
        """
            )
        model = model.to(device)

        # optimization if necessary -> auxiliary function
        model = pt_optimize_model(config, model, verbose)

        arg_package.update(
            {
                "model": model,
                "device": device,
                "use_gpu": config["use_gpu"],
                "dtype": dtype,
            }
        )
    config.update({"model_type": model_type, "model_args": arg_package})

    print(f"""    [ ] warming up the model...""")
    warmup(model_type, config, arg_package)

    return config


def prepare_output(
    config: Config,
    profile: dict,
    identifier: str = "",
) -> tuple[rasterio.io.DatasetWriter, str]:  # type: ignore
    """Prepare output raster profile and output path"""

    config, path_out = setup_indiv_path(config, identifier)
    size = config["img_pixels_detection"]

    out_profile = profile.copy()
    out_profile.update(
        {
            "dtype": "uint16",
            "compress": "LZW",
            "driver": "GTiff",
            "BIGTIFF": "YES",
            "tiled": True,
            "blockxsize": size,
            "blockysize": size,
        }
    )

    output_type = config["effective_output_type"]

    out_profile["count"] = 2 if output_type == "argmax" else config["n_classes"]
    # second band gives the max probability

    out = rasterio.open(path_out, "w+", **out_profile)
    return out, path_out


def prepare_data(
    config: Config,
) -> tuple[Sliced_Dataset, DataLoader, GeoDataFrame, dict]:

    stride = config["stride"]
    # slicing
    sliced_dataframe, profile, resolution = prepare_tiles(config, stride)

    # get dataset
    dataset = Sliced_Dataset(
        dataframe=sliced_dataframe,
        img_path=config["input_img_path"],
        resolution=resolution,
        bands=config["channels"],
        patch_detection_size=config["img_pixels_detection"],
        norma_dict=config["norma_task"],
    )

    # get Dataloader
    print(f"Using batch size: {config['batch_size']}")

    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_worker"],
        pin_memory=True,
    )

    return dataset, data_loader, sliced_dataframe, profile


def prepare_tiles(
    config: Config,
    stride: int,
) -> tuple[GeoDataFrame, dict, tuple[float, float]]:
    """Slicing extent for overlapping detection"""
    input_path = Path(config["input_img_path"])
    patch_size = config["img_pixels_detection"]
    margin = config["margin"]
    output_name = config["output_name"]
    output_path = Path(config["local_out"])
    write_df = config.get("write_dataframe", False)

    sliced_dataframe, profile, resolution, img_size = slice_extent(
        in_img=input_path,
        patch_size=patch_size,
        margin=margin,
        output_name=output_name,
        output_path=output_path,
        write_dataframe=write_df,
        stride=stride,
    )
    ## log
    log_verbose = config.get("log_verbose", True)
    if log_verbose:
        conf_log(config, resolution, img_size)
    print(f"""    [x] sliced input raster to {len(sliced_dataframe)} squares...""")

    return sliced_dataframe, profile, resolution
