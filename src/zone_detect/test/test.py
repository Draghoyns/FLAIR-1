from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.zone_detect.test.metrics import *
from src.zone_detect.utils import read_config
from src.zone_detect.test.tests import *


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

    config_dir = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/"

    # default
    # config_file = "config_argmax_small_irc_test.yaml"

    # comparison
    # config_file = "config-compare_strat.yaml"

    # metrics
    config_file = "config_detect_compare_metrics.yaml"

    config = config_dir + config_file

    # toy "image"
    img_size = 5, 5
    patch_size = 3
    margin = 0
    stride = 2
    query = (0, 5, 0, 5)

    """# prediction files
    # get a folder inside output_path
    list_preds = []
    pred_folder = Path(config["output_path"])
    pred_path = pred_folder
    for pred in pred_folder.iterdir():
        if not pred.is_dir():
            continue
        pred_path = pred

    for pred in pred_path.iterdir():
        if not pred.is_file() or not pred.name.endswith(".tif"):
            continue
        list_preds.append(pred)"""

    """# metrics analysis

    metrics_path = config["metrics_out"]

    metrics_data = load_metrics_json(metrics_path)
    df = flatten_metrics(metrics_data)

    param = "patch size"
    metrics = ["mIoU", "Overall Accuracy", "Fscore"]

    plot_metrics(analyze_param(df, param, metrics[0]), param, metrics[0])
"""

    # test error rate on margin 0

    truth = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/labels_raster/FLAIR_19/D037_2021"

    out_dir = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/swin-upernet-small/D037_2021/out20250515/batch_error"

    pred_dir = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/swin-upernet-small/D037_2021/out20250512"
    # "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/swin-upernet-small/D037_2021/out20250514/20250514_174851_margin=0"

    # test error rate
    # error_rate(truth, out_dir, pred)
    # test error rate on patch

    error_rate_loop(Path(truth), Path(out_dir), Path(pred_dir))
