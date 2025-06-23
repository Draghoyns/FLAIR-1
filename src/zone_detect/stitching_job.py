import math
import geopandas as gpd
import numpy as np

from typing import Any

from rasterio.features import geometry_window
from rasterio.io import DatasetWriter
from rasterio.windows import Window

from src.zone_detect.dataset import convert
from src.zone_detect.slicing_job import create_polygon_from_bounds

from src.zone_detect.test.tiles import (
    patch_overlap,
    patch_weights,
    total_weights,
    out_of_bounds,
)


def round_shape(window, op="ceil", pixel_precision=4) -> Window:
    """Rounds the width and height of a rasterio window manually."""
    if op not in {"ceil", "floor", "round"}:
        raise ValueError(f"Unsupported op: {op}")

    round_func = {"ceil": math.ceil, "floor": math.floor, "round": round}[op]

    # Get width and height
    width = window.width
    height = window.height

    # Apply rounding
    factor = 10**pixel_precision
    rounded_width = round_func(width * factor) / factor
    rounded_height = round_func(height * factor) / factor

    # Window is a frozen dataclass, so create a new instance with the desired values using from_slices
    row_slice = slice(int(window.row_off), int(window.row_off + rounded_height))
    col_slice = slice(int(window.col_off), int(window.col_off + rounded_width))
    return Window.from_slices(
        row_slice, col_slice, height=rounded_height, width=rounded_width
    )


def stitching(
    config: dict[str, Any],
    sliced_dataframe: gpd.GeoDataFrame,
    prediction: np.ndarray,
    index: np.ndarray,
    out: DatasetWriter,
    stitch: str,
    stride: int,
) -> tuple[np.ndarray, Window]:
    """Outputs patch handled, ready to be written"""

    margin = config["margin"]  # only for clipping
    img_size = config["img_pixels_detection"]
    output_type = config["output_type"]  # we only handle argmax for now

    i = index[0]
    sliced_box = [
        sliced_dataframe.at[i, "left"],
        sliced_dataframe.at[i, "right"],
        sliced_dataframe.at[i, "bottom"],
        sliced_dataframe.at[i, "top"],
    ]  # geo

    # align to resolution
    sliced_box = [round(coord, 3) for coord in sliced_box]

    if stitch == "exact-clipping" or output_type == "class_prob":
        # default
        # removing margins
        prediction = prediction[
            :,
            0 + margin : img_size - margin,
            0 + margin : img_size - margin,
        ]
        prediction = convert(prediction, output_type)

        # get the window
        sliced_patch_bounds = create_polygon_from_bounds(*sliced_box)
        window = geometry_window(out, [sliced_patch_bounds], pixel_precision=6)
        window = round_shape(window, op="ceil", pixel_precision=4)
        return prediction, window

    else:

        # _________GETTING_WINDOW__________#
        # out of bounds handling and get the patch plus the margin

        i = index[0]
        bigbox = [
            sliced_dataframe.at[i, "left_o"],
            sliced_dataframe.at[i, "right_o"],
            sliced_dataframe.at[i, "bottom_o"],
            sliced_dataframe.at[i, "top_o"],
        ]  # geo
        oob = np.array(out_of_bounds(bigbox, sliced_box)).astype(int)
        oob[0] *= -1
        oob[2] *= -1

        bounding_box = np.array(sliced_box) + oob * margin  # geo

        window = geometry_window(
            out, [create_polygon_from_bounds(*bounding_box)], pixel_precision=6
        )

        possible_overlap = out.read(
            window=window
        )  # array of shape (bands, height, width)

        # help averaging
        size = out.profile["width"], out.profile["height"]
        # yes be careful, sliced_box should be in pixel coord for this function
        overlapping = patch_overlap(size, img_size, sliced_box, stride)

        # note : be really careful where you have geo coord and pixel coord
        # TODO : stay at pixel level the longest possible

        if stitch == "average":  # only for class_prob

            prediction = prediction / overlapping
            prediction += possible_overlap
            prediction = convert(prediction, output_type)

            pass
        elif stitch == "average_weights":
            weights = patch_weights(img_size, sigma=0.5, mode="exp")
            distance_map = total_weights(size, img_size, sliced_box, stride)
            prediction = prediction * weights / distance_map
            prediction += possible_overlap
            prediction = convert(prediction, output_type)

        elif stitch == "max":
            prediction = convert(prediction, output_type)

            better_past = possible_overlap[0] > prediction[0]
            prediction[:, better_past] = possible_overlap[:, better_past]

    return prediction, window
