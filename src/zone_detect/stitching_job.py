import math
import geopandas as gpd
import numpy as np

from typing import Any

from rasterio.features import geometry_window
from rasterio.io import DatasetWriter
from rasterio.windows import Window

from src.zone_detect.dataset import convert
from src.zone_detect.slicing_utils import (
    geo_patch_overlap,
    patch_weights,
    total_weights,
    out_of_bounds,
    create_polygon_from_bounds,
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
    param_combi: dict[str, Any],
    sliced_dataframe: gpd.GeoDataFrame,
    prediction: np.ndarray,
    index: np.ndarray,
    out: DatasetWriter,
    output_type: str,
) -> tuple[np.ndarray, Window]:
    """Outputs patch handled, ready to be written.
    Args:
        param_combi: dict containing the parameters for stitching
            - stitching, margin, img_pixels_detection, stride
            - stitching supports 'exact-clipping', 'average', 'average-weights', 'max'
        sliced_dataframe: GeoDataFrame with the sliced patches
        prediction: model's prediction for the patch
        index: index of the patch in the sliced dataframe
        out: rasterio DatasetWriter to write the output
        output_type: type of output to convert to (e.g., 'argmax', 'class_prob')
    Returns:
        tuple of prediction and window to write

    TODO: custom function for overlap handling
    """

    margin = param_combi["margin"]
    img_size = param_combi["img_pixels_detection"]
    stitch = param_combi["stitching"]
    stride = param_combi["stride"]

    # get inner box coordinates
    i = index[0]
    effective_patch_geo_box = [
        sliced_dataframe.at[i, "left"],
        sliced_dataframe.at[i, "right"],
        sliced_dataframe.at[i, "bottom"],
        sliced_dataframe.at[i, "top"],
    ]  # geo

    # align to resolution
    effective_patch_geo_box = [round(coord, 3) for coord in effective_patch_geo_box]

    if "exact" in stitch:
        # default
        prediction = exact_stitching(prediction, img_size, margin, output_type)

        # get the window
        eff_patch_bounds = create_polygon_from_bounds(*effective_patch_geo_box)
        window = geometry_window(out, [eff_patch_bounds], pixel_precision=6)
        window = round_shape(window, op="ceil", pixel_precision=4)

        return prediction, window

    else:

        # _________GETTING_WINDOW__________#
        # out of bounds handling and get the patch plus the margin

        bigbox = [
            sliced_dataframe.at[i, "left_o"],
            sliced_dataframe.at[i, "right_o"],
            sliced_dataframe.at[i, "bottom_o"],
            sliced_dataframe.at[i, "top_o"],
        ]  # geo, whole image bounds

        bigbox = [round(coord, 3) for coord in bigbox]

        # don't take margin into account for out of bounds
        oob = np.array(out_of_bounds(bigbox, effective_patch_geo_box)).astype(int)
        oob[0] *= -1
        oob[2] *= -1

        resolution = abs(round(sliced_dataframe.at[i, "resolution_x"], 5))

        geomargin = margin * resolution

        patch_including_margin = (
            np.array(effective_patch_geo_box) + oob * geomargin
        )  # geo

        window = geometry_window(
            out,
            [create_polygon_from_bounds(*patch_including_margin)],
            pixel_precision=6,
        )

        possible_overlap = out.read(
            window=window
        )  # array of shape (bands, patch_size, patch_size)

        # note : be really careful where you have geo coord and pixel coord
        # TODO : stay at pixel level the longest possible

        if stitch == "average":

            prediction = average_stitching(
                prediction,
                possible_overlap,
                img_size,
                patch_including_margin,
                bigbox,
                stride,
                margin,
                out,
                oob,
            )

        elif stitch == "average-weights":
            weights = patch_weights(img_size, sigma=0.5, mode="exp")
            size = out.profile["width"], out.profile["height"]
            distance_map = total_weights(
                size, img_size, effective_patch_geo_box, stride
            )
            prediction = prediction * weights / distance_map
            prediction += possible_overlap

        prediction = convert(prediction, output_type)

        if stitch == "max":
            better_past = possible_overlap[0] > prediction[0]
            prediction[:, better_past] = possible_overlap[:, better_past]

    return prediction, window


def average_stitching(
    prediction: np.ndarray,
    possible_overlap: np.ndarray,
    img_size: int,
    bounding_box: list[float],
    bigbox: list[float],
    stride: int,
    margin: int,
    out: DatasetWriter,
    oob: np.ndarray,
) -> np.ndarray:
    """Average the prediction with the overlapping patch."""

    overlapping = geo_patch_overlap(out, img_size, bounding_box, bigbox, stride)

    h, w = possible_overlap.shape[-2:]
    # If oob[0] (left) == -1, crop from left side
    col_start = margin if oob[0] == -1 else 0
    col_end = col_start + w
    # If oob[2] (bottom) == -1, crop from bottom side
    row_start = margin if oob[2] == -1 else 0
    row_end = row_start + h

    # crop of bounds elements
    prediction = prediction[:, row_start:row_end, col_start:col_end]

    print(f"\nFirst pixel gives the prediction:\n {prediction[:, 0, 0]}")

    prediction = prediction / overlapping

    print(f"\nFirst pixel after overlap handling:\n {prediction[:, 0, 0]}")

    prediction += possible_overlap

    print(f"\nFirst pixel after adding current result:\n {prediction[:, 0, 0]}")

    return prediction


def exact_stitching(
    prediction: np.ndarray,
    img_size: int,
    margin: int,
    output_type: str,
) -> np.ndarray:
    """Perform exact stitching by removing margins."""

    # removing margins
    prediction = prediction[
        :,
        0 + margin : img_size - margin,
        0 + margin : img_size - margin,
    ]

    prediction = convert(prediction, output_type)

    return prediction
