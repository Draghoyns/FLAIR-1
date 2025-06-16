import os
from pathlib import Path

import numpy as np
import geopandas as gpd

import rasterio

from shapely import Polygon
from shapely.geometry import box, mapping
from typing import Tuple


PixCoord = Tuple[int, int, int, int]


def create_box_from_bounds(
    x_min: float, x_max: float, y_min: float, y_max: float
) -> Polygon:
    return box(x_min, y_max, x_max, y_min)


def create_polygon_from_bounds(
    x_min: float, x_max: float, y_min: float, y_max: float
) -> dict:
    return mapping(box(x_min, y_max, x_max, y_min))


def slice_extent(
    in_img: Path,
    patch_size: int,
    margin: int,
    output_path: Path,
    output_name: str,
    write_dataframe: bool,
    stride: int,
) -> tuple[gpd.GeoDataFrame, dict, tuple[float, float], tuple[int, int]]:

    with rasterio.open(in_img) as src:
        img_width, img_height = src.read(1).shape
        profile = src.profile
        min_x, min_y, max_x, max_y = src.bounds
        resolution_x, resolution_y = map(lambda r: abs(round(r, 5)), src.res)

    # geo conversion
    geo_output_w, geo_output_h = patch_size * resolution_x, patch_size * resolution_y
    geo_margin_x, geo_margin_y = margin * resolution_x, margin * resolution_y

    if stride:
        geo_step = [stride * resolution_x, stride * resolution_y]
    else:  # default
        geo_step = [
            geo_output_w - (2 * geo_margin_x),
            geo_output_h - (2 * geo_margin_y),
        ]

    # initializing
    tmp_list = []
    geo_patches = set()  # To track unique patches

    X = np.arange(min_x - geo_margin_x, max_x + geo_margin_x, geo_step[0])
    Y = np.arange(min_y - geo_margin_y, max_y + geo_margin_y, geo_step[1])

    for x_coord in X:

        # Adjust last column to ensure proper alignment
        if x_coord + geo_output_w > max_x + geo_margin_x:
            x_coord = max_x + geo_margin_x - geo_output_w

        for y_coord in Y:
            # Adjust last row
            if y_coord + geo_output_h > max_y + geo_margin_y:
                y_coord = max_y + geo_margin_y - geo_output_h

            # Define patch boundaries, geo, absolute position
            # Ensure patches don't go outside raster bounds
            left = x_coord + geo_margin_x
            right = min(x_coord + geo_output_w - geo_margin_x, max_x)
            bottom = y_coord + geo_margin_y
            top = min(y_coord + geo_output_h - geo_margin_y, max_y)

            col, row = (
                int((y_coord - min_y) // resolution_x) + 1,
                int((x_coord - min_x) // resolution_y) + 1,
            )

            # Unique identifier for patch
            new_patch = (
                round(left, 6),
                round(bottom, 6),
                round(right, 6),
                round(top, 6),
            )

            if new_patch not in geo_patches:
                geo_patches.add(new_patch)  # Track unique patches
                row_d = {
                    "id": str(f"{1}-{row}-{col}"),
                    "output_id": output_name,
                    "job_done": 0,
                    "left": left,
                    "bottom": bottom,
                    "right": right,
                    "top": top,
                    "left_o": min_x,
                    "bottom_o": min_y,
                    "right_o": max_x,
                    "top_o": max_y,
                    "geometry": create_box_from_bounds(
                        x_coord,
                        x_coord + geo_output_w,
                        y_coord,
                        y_coord + geo_output_h,
                    ),
                }
                tmp_list.append(row_d)

    gdf_output = gpd.GeoDataFrame(tmp_list, crs=profile["crs"], geometry="geometry")

    if write_dataframe:
        gdf_output.to_file(
            os.path.join(
                output_path, output_name.split(".tif")[0] + "_slicing_job.gpkg"
            ),
            driver="GPKG",
        )

    return gdf_output, profile, (resolution_x, resolution_y), (img_width, img_height)


def slice_extent_separate(
    in_img: Path,
    patch_size: int,
    margin: int,
    output_path: Path,
    output_name: str,
    write_dataframe: bool,
    stride: int,
) -> tuple[gpd.GeoDataFrame, dict, tuple[float, float], tuple[int, int]]:
    """It sucks because there is a slight shift of pixel, making the metrics evaluation wrong"""

    img_size = rasterio.open(in_img).read(1).shape[::-1]  # (width, height)
    patches = slice_pixels(img_size, patch_size, margin, stride)

    geo_slices = slice_geo(
        in_img, margin, output_path, output_name, write_dataframe, patches
    )

    return geo_slices


def slice_geo(
    in_img: Path,
    margin: int,
    output_path: Path,
    output_name: str,
    write_dataframe: bool,
    patches: list[tuple[int, int, int, int]],
) -> tuple[gpd.GeoDataFrame, dict, tuple[float, float], tuple[int, int]]:

    # get geo info
    with rasterio.open(in_img) as src:
        profile = src.profile
        img_width, img_height = profile["width"], profile["height"]
        left_overall, bottom_overall, right_overall, top_overall = src.bounds
        resolution = abs(round(src.res[0], 5)), abs(round(src.res[1], 5))

    # geo conversion
    geo_margin = [margin * resolution[0], margin * resolution[1]]

    min_x, min_y = left_overall, bottom_overall
    max_x, max_y = right_overall, top_overall

    # initializing
    tmp_list = []
    geo_patches = set()

    # get the slicing
    pixel_patches = patches

    for patch in pixel_patches:
        x_min_patch, x_max_patch, y_min_patch, y_max_patch = patch
        # patch without the margin

        # geo conversion, small patch
        left = x_min_patch * resolution[0] + min_x
        right = x_max_patch * resolution[0] + min_x
        bottom = y_min_patch * resolution[1] + min_y
        top = y_max_patch * resolution[1] + min_y

        # big patch (add margin)
        left_patch = left - geo_margin[0]
        right_patch = right + geo_margin[0]
        bottom_patch = bottom - geo_margin[1]
        top_patch = top + geo_margin[1]

        # Ensure patches don't go outside raster bounds after adding margins
        right = min(right, max_x)
        top = min(top, max_y)

        # Unique identifier for patch
        col, row = (
            int((left_patch - min_x) // resolution[0]) + 1,
            int((bottom_patch - min_y) // resolution[1]) + 1,
        )
        new_patch = (
            round(left, 6),
            round(bottom, 6),
            round(right, 6),
            round(top, 6),
        )
        if new_patch not in geo_patches:
            geo_patches.add(new_patch)  # Track unique patches
            row_d = {
                "id": f"1-{row}-{col}",
                "output_id": output_name,
                "job_done": 0,
                "left": left,
                "bottom": bottom,
                "right": right,
                "top": top,
                "left_o": left_overall,
                "bottom_o": bottom_overall,
                "right_o": right_overall,
                "top_o": top_overall,
                "geometry": create_box_from_bounds(
                    left_patch,
                    right_patch,
                    bottom_patch,
                    top_patch,
                ),
            }
            tmp_list.append(row_d)

    gdf_output = gpd.GeoDataFrame(tmp_list, crs=profile["crs"], geometry="geometry")

    if write_dataframe:
        gdf_output.to_file(
            os.path.join(
                output_path, output_name.split(".tif")[0] + "_slicing_job.gpkg"
            ),
            driver="GPKG",
        )

    return gdf_output, profile, resolution, (img_width, img_height)


def slice_pixels(
    img_size: tuple[int, int],
    patch_size: int,
    margin: int,
    stride: int,
) -> list[PixCoord]:
    """
    Generate patches for a given image size.
    The patches are the small boxes where the margins were removed.
    They will be added for inference inside the slice_geo function."""

    def _add_patch_if_valid(patches: set[PixCoord], x_min: int, y_min: int):
        x_max = x_min + patch_size
        y_max = y_min + patch_size
        if x_max <= x_size and y_max <= y_size:
            patches.add((x_min, x_max, y_min, y_max))
        return patches

    patches = set()
    x_size, y_size = img_size

    patch_size = patch_size - 2 * margin

    for y in range(0, y_size + 1, stride):
        for x in range(0, x_size + 1, stride):
            # bottom right corner
            patches = _add_patch_if_valid(patches, x, y)

    # add edge cases
    if y_size - patch_size > 0 and (y_size - patch_size) % stride != 0:
        # bottom
        y = y_size - patch_size
        for x in range(0, x_size - patch_size + 1, stride):
            patches = _add_patch_if_valid(patches, x, y)

    if x_size - patch_size > 0 and (x_size - patch_size) % stride != 0:
        # right
        x = x_size - patch_size
        for y in range(0, y_size - patch_size + 1, stride):
            patches = _add_patch_if_valid(patches, x, y)

        # add the last patch
    if (
        y_size - patch_size > 0
        and (y_size - patch_size) % stride != 0
        and x_size - patch_size > 0
        and (x_size - patch_size) % stride != 0
    ):
        y = y_size - patch_size
        x = x_size - patch_size
        patches = _add_patch_if_valid(patches, x, y)

    return sorted(patches)


def nb_patches(
    img_size: tuple[int, int],
    stride: int,
) -> int:
    """
    Calculate the number of patches for a given image size.
    Lightweight version of slice_pixels that does not generate patches.
    """

    x_size, y_size = img_size

    # Calculate the number of patches in each dimension
    inexact_border_x = int(x_size % stride != 0)
    inexact_border_y = int(y_size % stride != 0)

    num_patches_x = x_size // stride + inexact_border_x
    num_patches_y = y_size // stride + inexact_border_y

    total_patches = num_patches_x * num_patches_y
    if inexact_border_x and inexact_border_y:
        # get rid of the last overlapping patch
        total_patches = total_patches - 1

    return total_patches
