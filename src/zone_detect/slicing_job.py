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
    in_img: Path,
    patch_size: int,
    margin: int,
    output_path: Path,
    output_name: str,
    write_dataframe: bool,
    stride: int,
) -> tuple[gpd.GeoDataFrame, dict, tuple[float, float], list[int]]:

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

    return gdf_output, profile, (resolution_x, resolution_y), [img_width, img_height]


def slice_extent_separate(
    in_img: Path,
    patch_size: int,
    margin: int,
    output_path: Path,
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
