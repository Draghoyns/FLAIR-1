from pathlib import Path
from geopandas import GeoDataFrame
import rasterio

from torch.utils.data import DataLoader

from typing import Any

from src.zone_detect.dataset import Sliced_Dataset

from src.zone_detect.slicing_job import slice_extent
from src.zone_detect.utils import conf_log, setup_indiv_path

Config = dict[str, Any]


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

    output_type = config.get("effective_output_type", "argmax")

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
