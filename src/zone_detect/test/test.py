import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio

from src.zone_detect.test.metrics import valid_truth
from src.zone_detect.test.tiles import get_stride
from src.zone_detect.utils import read_config
from src.zone_detect.test.pixel_operation import slice_pixels


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


def error_rate_patch(config: dict, out_dir: str, pred_filename: str = "") -> None:
    # output file
    assert out_dir is not None, "Please provide an output path for the metrics"

    truth_path = valid_truth(config)

    # prediction path
    pred_folder = Path(config["output_path"])
    pred_path = pred_folder / pred_filename
    if pred_filename == "":
        pred_path = next(pred_folder.glob("*.tif"), None)  # select the first one
    if pred_path is None:
        raise ValueError(f"No prediction file found in {pred_folder}")

    # load images in arrays
    # PIL struggles (multi band, 16 bit, float32)
    with rasterio.open(truth_path) as src:
        target = src.read(1) - 1  # -1 to match the prediction
    with rasterio.open(pred_path) as src:
        pred = src.read(1)

    # slice into patches
    img_size = target.shape[0], target.shape[1]
    patches = slice_pixels(
        img_size,
        config["img_pixels_detection"],
        config["margin"],
        get_stride(config)[0],
    )
    effective_patch_size = config["img_pixels_detection"] - 2 * config["margin"]
    out_array = np.zeros(
        (effective_patch_size, effective_patch_size),
    )
    # iterate over the patches and access images using patches indices ?
    for patch in patches:
        bottom, top, left, right = patch

        target_patch = target[bottom:top, left:right]
        pred_patch = pred[bottom:top, left:right]

        # compute the error rate
        # for each pixel, increment if not equal
        # hard error rate
        # replace / evaluate soft confidence
        out_array += np.where(target_patch != pred_patch, 1, 0)

    out_array = out_array / len(patches)

    # i'm afraid of the memory

    # useful info : method used, error rate

    method = str(pred_path).split("ARGMAX-IRC-S_")[-1].split(".tif")[0]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = out_path / f"error_rate_{method}_{datetime.datetime.now()}.png"

    # save the error rate as a png
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(out_array, cmap="hot", interpolation="nearest", vmin=0.025, vmax=0.25)
    plt.colorbar()
    plt.title("Error Rate for method : \n" + method)
    plt.savefig(str(out_path))
    plt.close()
    print(f"Error rate saved to {out_path}")


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

    # read config
    config = read_config(config)

    # prediction files
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
        list_preds.append(pred)

    for pred in list_preds:
        sub_file = str(pred).split("/")[-2] + "/" + pred.name
        # error rate
        error_rate_patch(
            config,
            config["output_path"],
            sub_file,
        )
