import os
from pathlib import Path
import geopandas as gpd
import rasterio
from shapely import Polygon
from shapely.geometry import box


def create_box_from_bounds(
    x_min: float, x_max: float, y_min: float, y_max: float
) -> Polygon:
    return box(x_min, y_max, x_max, y_min)


def slice_geo(
    in_img: Path,
    margin: int,
    output_path: Path,
    output_name: str,
    write_dataframe: bool,
    patches: list[tuple[int, int, int, int]],
) -> tuple[gpd.GeoDataFrame, dict, tuple[float, float], list[int]]:

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

    return gdf_output, profile, resolution, [img_width, img_height]
