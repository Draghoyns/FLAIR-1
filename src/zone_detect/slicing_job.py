import os
import numpy as np
import rasterio
import geopandas as gpd

from pathlib import Path
from shapely.geometry import box, mapping

from src.zone_detect.test.geo_operation import slice_geo, create_box_from_bounds
from src.zone_detect.test.pixel_operation import slice_pixels


def create_polygon_from_bounds(
    x_min: float, x_max: float, y_min: float, y_max: float
) -> dict:
    return mapping(box(x_min, y_max, x_max, y_min))


def slice_extent(
    in_img: str | Path,
    patch_size: int,
    margin: int,
    output_path: str | Path,
    output_name: str,
    write_dataframe: bool,
    stride: int,
) -> tuple[gpd.GeoDataFrame, dict, tuple[float, float], list[int]]:

    with rasterio.open(in_img) as src:
        img_width, img_height = src.read(1).shape
        profile = src.profile
        left_overall, bottom_overall, right_overall, top_overall = src.bounds
        resolution = abs(round(src.res[0], 5)), abs(round(src.res[1], 5))

    # geo conversion
    geo_output_size = [patch_size * resolution[0], patch_size * resolution[1]]
    geo_margin = [margin * resolution[0], margin * resolution[1]]

    if stride:
        geo_step = [stride * resolution[0], stride * resolution[1]]
    else:  # default
        geo_step = [
            geo_output_size[0] - (2 * geo_margin[0]),
            geo_output_size[1] - (2 * geo_margin[1]),
        ]

    min_x, min_y = left_overall, bottom_overall
    max_x, max_y = right_overall, top_overall

    # initializing
    tmp_list = []
    geo_patches = set()  # To track unique patches

    X = np.arange(min_x - geo_margin[0], max_x + geo_margin[0], geo_step[0])
    Y = np.arange(min_y - geo_margin[1], max_y + geo_margin[1], geo_step[1])

    for x_coord in X:
        for y_coord in Y:
            # for each patch

            # Adjust last column to ensure proper alignment
            if x_coord + geo_output_size[0] > max_x + geo_margin[0]:
                x_coord = max_x + geo_margin[0] - geo_output_size[0]
            # Adjust last row
            if y_coord + geo_output_size[1] > max_y + geo_margin[1]:
                y_coord = max_y + geo_margin[1] - geo_output_size[1]

            # Define patch boundaries, geo, absolute position
            left = x_coord + geo_margin[0]
            right = x_coord + geo_output_size[0] - geo_margin[0]
            bottom = y_coord + geo_margin[1]
            top = y_coord + geo_output_size[1] - geo_margin[1]

            # Ensure patches don't go outside raster bounds
            right = min(right, max_x)
            top = min(top, max_y)

            col, row = (
                int((y_coord - min_y) // resolution[0]) + 1,
                int((x_coord - min_x) // resolution[1]) + 1,
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
                    "left_o": left_overall,
                    "bottom_o": bottom_overall,
                    "right_o": right_overall,
                    "top_o": top_overall,
                    "geometry": create_box_from_bounds(
                        x_coord,
                        x_coord + geo_output_size[0],
                        y_coord,
                        y_coord + geo_output_size[1],
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

    return gdf_output, profile, resolution, [img_width, img_height]


def slice_extent_separate(
    in_img: str | Path,
    patch_size: int,
    margin: int,
    output_path: str | Path,
    output_name: str,
    write_dataframe: bool,
    stride: int,
) -> tuple[gpd.GeoDataFrame, dict, tuple[float, float], list[int]]:
    """It sucks because there is a slight shift of pixel, making the metriucs evaluation wrong"""

    img_size = rasterio.open(in_img).read(1).shape[::-1]  # (width, height)
    patches = slice_pixels(img_size, patch_size, margin, stride)

    geo_slices = slice_geo(
        in_img, margin, output_path, output_name, write_dataframe, patches
    )

    return geo_slices
