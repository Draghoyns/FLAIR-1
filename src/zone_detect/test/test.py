from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from src.zone_detect.test.tiles import patch_weights
from src.zone_detect.test.pixel_operation import slice_pixels
from src.zone_detect.test.visualization import viz_slicing


def geogr_patches(
    in_img: str | Path,
    patches: list[tuple[int, int, int, int]],
) -> list[tuple[float, float, float, float]]:
    """Generate patches for a given image size."""

    # geo conversion
    geo_patches = []

    with rasterio.open(in_img) as src:
        profile = src.profile
        left_overall, bottom_overall, right_overall, top_overall = src.bounds
        resolution = abs(round(src.res[0], 5)), abs(round(src.res[1], 5))

    for patch in patches:
        x_min_patch, x_max_patch, y_min_patch, y_max_patch = patch
        # patch without the margin

        # geo conversion
        left = x_min_patch * resolution[0]
        right = x_max_patch * resolution[0]
        bottom = y_min_patch * resolution[1]
        top = y_max_patch * resolution[1]

        L = left_overall + left
        R = left_overall + right
        B = bottom_overall - bottom
        T = bottom_overall - top

        geo_patches.append((L, B, R, T))

        # debug time

        # if out of bounds, print the coordinates
        if (
            L < left_overall
            or R > right_overall
            or B < bottom_overall
            or T > top_overall
        ):
            print(
                f"Patch out of bounds: {L}, {B}, {R}, {T} "
                f"({left_overall}, {bottom_overall}, {right_overall}, {top_overall})"
            )

    return geo_patches


def ground_truth_geoconv(
    in_img: str | Path, patch_size: int, margin: int, stride: int
) -> list[tuple[float, float, float, float]]:
    with rasterio.open(in_img) as src:
        left_overall, bottom_overall, right_overall, top_overall = src.bounds
        resolution = abs(round(src.res[0], 5)), abs(round(src.res[1], 5))

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

    X = np.arange(min_x - geo_margin[0], max_x + geo_margin[0], geo_step[0])
    Y = np.arange(min_y - geo_margin[1], max_y + geo_margin[1], geo_step[1])

    geo_patches = []
    for x_coord in X:
        for y_coord in Y:
            # for each patch

            # Adjust last column to ensure proper alignment
            if x_coord + geo_output_size[0] > max_x + geo_margin[0]:
                x_coord = max_x + geo_margin[0] - geo_output_size[0]
            # Adjust last row
            if y_coord + geo_output_size[1] > max_y + geo_margin[1]:
                y_coord = max_y + geo_margin[1] - geo_output_size[1]

            # Define patch boundaries, geo
            left = x_coord + geo_margin[0]
            right = x_coord + geo_output_size[0] - geo_margin[0]
            bottom = y_coord + geo_margin[1]
            top = y_coord + geo_output_size[1] - geo_margin[1]

            # Ensure patches don't go outside raster bounds
            right = min(right, max_x)
            top = min(top, max_y)

            geo_patches.append((left, bottom, right, top))

            # debug time

            # if out of bounds, print the coordinates
            if (
                left < left_overall
                or right > right_overall
                or bottom < bottom_overall
                or top > top_overall
            ):
                print(
                    f"Patch out of bounds: {left}, {bottom}, {right}, {top} "
                    f"({left_overall}, {bottom_overall}, {right_overall}, {top_overall})"
                )

    return geo_patches


# example

if __name__ == "__main__":
    img_size = 5, 5
    patch_size = 3
    margin = 0
    stride = 2
    query = (0, 5, 0, 5)

    # visualize
    patches = slice_pixels(img_size, patch_size, margin, stride)
