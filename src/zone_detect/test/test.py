import os
from pathlib import Path
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt

from src.zone_detect.slicing_job import create_box_from_bounds
from src.zone_detect.test.tiles import patch_weights, total_weights


def slice_pixels(
    img_size: tuple[int, int],
    patch_size: int,
    margin: int,
    stride: int,
) -> set[tuple[int, int, int, int]]:

    def _add_patch_if_valid(patches, x_min, y_min):
        x_max = x_min + patch_size
        y_max = y_min + patch_size
        if x_max <= x_size and y_max <= y_size:
            patches.add((x_min, x_max, y_min, y_max))
        return patches

    patches = set()
    x_size, y_size = img_size

    # we want the patches that will be put into the GeoDataFrame (where we took out the margin)
    # the margin will be added for the geometry inside the slice_geo

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

    return patches


def slice_geo(
    in_img: str | Path,
    patch_size: int,
    margin: int,
    stride: int,
    output_path: str | Path,
    output_name: str,
    write_dataframe: bool,
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
    pixel_patches = slice_pixels(
        img_size=(img_width, img_height),
        patch_size=patch_size,
        margin=margin,
        stride=stride,
    )

    for patch in pixel_patches:
        x_min_patch, x_max_patch, y_min_patch, y_max_patch = patch

        # geo conversion and big patch (add margin)
        left_patch = x_min_patch * resolution[0] + geo_margin[0]
        right_patch = x_max_patch * resolution[0] - geo_margin[0]
        bottom_patch = y_min_patch * resolution[1] + geo_margin[1]
        top_patch = y_max_patch * resolution[1] - geo_margin[1]

        # absolute position
        left = min_x + left_patch
        right = min_x + right_patch
        bottom = min_y + bottom_patch
        top = min_y + top_patch

        # Ensure patches don't go outside raster bounds after adding margins
        right = min(right, max_x)
        top = min(top, max_y)

        col, row = (
            int((bottom_patch - min_y) // resolution[0]) + 1,
            int((left_patch - min_x) // resolution[1]) + 1,
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


# example

if __name__ == "__main__":
    img_size = 5, 5
    patch_size = 3
    margin = 0
    stride = 2
    query = (0, 5, 0, 5)

    patches = slice_pixels(img_size, patch_size, margin, stride)

    local_weights = patch_weights(patch_size)

    # visualize

    plt.plot(local_weights)
    plt.title("Weights")
    plt.xlabel("Patch index")
    plt.ylabel("Weight")
    plt.show()
