import colorsys
import datetime
import gc
import json
import os
import psutil
from tqdm import tqdm

from adjustText import adjust_text

import lazy_import


wandb = lazy_import.lazy_module("wandb")

from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import rasterio
from rasterio.windows import Window

from sklearn.metrics import confusion_matrix

from src.zone_detect.model import (
    load_model,
)
from src.zone_detect.slicing_job import slice_pixels
from src.zone_detect.utils import extract_method, info_extract
from src.zone_detect.slicing_job import nb_patches

Config = dict[str, Any]  # type alias for configuration dictionary


#### UTILS ####
def clean_confmat(confmat: np.ndarray, config: Config) -> np.ndarray:
    #### CLEAN REGARDING WEIGHTS FOR METRICS CALC :
    weights = np.array([class_info[0] for class_info in config["classes"].values()])
    unused_classes = np.where(weights == 0)[0]
    if unused_classes.size > 0:
        confmat_cleaned = np.delete(confmat, unused_classes, axis=0)  # remove rows
        confmat_cleaned = np.delete(
            confmat_cleaned, unused_classes, axis=1
        )  # remove columns

        return confmat_cleaned
    return confmat


def valid_truth(config: Config) -> Path:
    """Check if the ground truth path is valid and coherent with the input path :
    the zone should be the same in both paths.
    For early error detection.
    """
    truth_path = Path(config["truth_path"])
    input_path = Path(config["input_img_path"])
    # verify coherence with input path
    truth_zone = truth_path.parts[-3:-1]
    input_zone = input_path.parts[-3:-1]
    if truth_zone != input_zone:
        raise ValueError(
            f"Ground truth path {truth_path} does not match input path {config['input_img_path']}"
        )
    return Path(truth_path)


def get_truth_path(pred_path: Path, truth_dir: Path) -> Path:
    """Get the corresponding ground truth path for a given prediction path. Prediction data file and ground truth data file are supposed containing the same zone information.

    For avoiding strict prediction / label couple passed as argument."""

    dpt, zone_name = info_extract(pred_path)["dpt"], info_extract(pred_path)["zone"]

    # corresponding ground truth
    # we consider gt_folder the overall folder
    truth_subdir = truth_dir / zone_name
    truth_path = next(truth_subdir.rglob("*.tif"), None)
    if truth_path is None:
        raise FileNotFoundError(
            f"Ground truth file not found in {truth_subdir}. Please check the folder."
        )
    return truth_path


def collect_paths_truth(config: Config, truth_dir: Path) -> pd.DataFrame:
    """Create DataFrame with prediction and ground truth paths matched from config. Predictions are supposed stored in output path specified in config.

    For metrics computation.
    """
    path_collection = []

    # get predictions
    if "local_out" not in config:
        pred_dir = Path(config["output_path"])
    else:
        pred_dir = Path(config["local_out"]).parent
        # manual fix, paths should be handled more cleanly
    timed_folders = [p for p in pred_dir.iterdir() if p.is_dir()]

    # dataframe with pred path, gt path and method single string
    for timestamp in sorted(timed_folders):
        pred_files = list(timestamp.rglob("*.tif"))

        truth_path = get_truth_path(pred_files[0], truth_dir)

        for pred_path in pred_files:
            method_id = info_extract(pred_path)["method"]
            # dpt_zone-name_data-type-ARGMAX-S_size=128_stride=96_margin=32_padding=some-padding_stitching=exact-clipping
            path_collection.append(
                {
                    "pred_path": str(pred_path),
                    "truth_path": str(truth_path),
                    "method": method_id,
                }
            )
    return pd.DataFrame(path_collection)


#### METRICS ####
def overall_accuracy(npcm: np.ndarray) -> float:
    total = npcm.sum()
    if total == 0:
        return 0.0

    return 100 * np.trace(npcm) / total


def class_IoU(npcm: np.ndarray) -> tuple[np.ndarray, float, float]:
    tp = np.diag(npcm)
    fn = np.sum(npcm, axis=1) - tp
    fp = np.sum(npcm, axis=0) - tp
    denom = tp + fn + fp

    ious = 100 * tp / denom
    ious = np.nan_to_num(ious)

    return ious, ious.mean(), ious.std()


def class_precision(npcm: np.ndarray) -> tuple[np.ndarray, float, float]:
    tp = np.diag(npcm)
    fp = np.sum(npcm, axis=0) - tp
    fn = np.sum(npcm, axis=1) - tp

    precision = tp / (tp + fp)
    precision = np.nan_to_num(precision)  # replaces NaN with 0

    return precision, precision.mean(), precision.std()


def class_recall(npcm: np.ndarray) -> tuple[np.ndarray, float, float]:
    tp = np.diag(npcm)
    fp = np.sum(npcm, axis=0) - tp
    fn = np.sum(npcm, axis=1) - tp

    recall = tp / (tp + fn)
    recall = np.nan_to_num(recall)  # replaces NaN with 0

    return recall, recall.mean(), recall.std()


def class_fscore(npcm: np.ndarray) -> tuple[np.ndarray, float, float]:
    tp = np.diag(npcm)
    fp = np.sum(npcm, axis=0) - tp
    fn = np.sum(npcm, axis=1) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    fscore = 2 * precision * recall / (precision + recall) * 100
    fscore = np.nan_to_num(fscore)  # replaces NaN with 0

    return fscore, fscore.mean(), fscore.std()


#### COMPUTATION ####
def compute_metrics_patch(
    pred_patch: np.ndarray,
    truth: np.ndarray,
    window: Window,
    config: Config,
    method: str,
) -> dict[str, Any]:
    """
    Patch metrics can be computed before the stitching ,
    or once the whole image is built.
    Average etrics are not exactly relevant because of the classes absent from a patch.
    Args:
        pred_patch (np.ndarray): Predicted patch.
        window (Window): Window object for the patch.
        config (dict): Configuration, in which the parameters for the inference are specified
        out_json (Path): Path to the output JSON file for metrics. If the file exists, it will be overwritten.
            You better put in the name if it's raw (before stitching) or after.
    """

    # raise error if invalid truth
    valid_truth(config)

    target = truth[
        window.row_off : window.row_off + window.height,
        window.col_off : window.col_off + window.width,
    ]

    # get the class predictions and remove the probabilities
    if target.shape != pred_patch.shape:
        pred_patch = pred_patch[0]

    classes = config["classes"]
    n_classes = len(classes)

    #### compute metrics
    # confusion matrix
    confmat = confusion_matrix(
        target.flatten(), pred_patch.flatten(), labels=range(n_classes)
    )

    confmat_cleaned = clean_confmat(confmat, config)

    with np.errstate(divide="ignore", invalid="ignore"):
        # nans are handled don't worry
        per_c_ious, avg_ious, std_ious = class_IoU(confmat_cleaned)
        ovr_acc = overall_accuracy(confmat_cleaned)
        per_c_fscore, avg_fscore, std_fscore = class_fscore(confmat_cleaned)

    # save metrics to a json file : raw or post-stitching
    key = f"{method}_{window.col_off}_{window.row_off}"
    metrics = {
        key: {
            "Avg_metrics_name": [
                "mIoU",
                "Overall Accuracy",
                "Fscore",
            ],
            "Avg_metrics": [
                avg_ious,
                ovr_acc,
                avg_fscore,
            ],
            "Std_dev_metrics": [
                std_ious,
                "undefined",
                std_fscore,
            ],
            "classes": [classes[i][1] for i in range(1, n_classes + 1)],
            "per_class_iou": list(per_c_ious),
            "per_class_fscore": list(per_c_fscore),
        }
    }
    return metrics


def add_confusion(
    pred_path: Path,
    truth_path: Path,
    sum_confmat: np.ndarray,
    n_classes: int,
    stride: int,
) -> np.ndarray:
    """Utility function to add confusion matrix for a single pair of prediction and ground truth images."""
    try:
        # loading
        with rasterio.open(pred_path) as src:
            preds = src.read(1)
        with rasterio.open(truth_path) as src:
            target = src.read(1) - 1

        # weighted confusion matrix
        input_img_size = target.shape  # (height, width)
        num_patches = nb_patches(input_img_size, stride)

        sum_confmat += num_patches * confusion_matrix(
            target.flatten(), preds.flatten(), labels=range(n_classes)
        )
    except Exception as e:
        print(f"Error processing {pred_path} and {truth_path}: {e}")
    return sum_confmat


def process_metrics(
    confmat: np.ndarray,
    config: Config,
    info: Config,
    method_times: dict,
    method_patches: list[int],
    area: float = 0.0,
    carbon: float = 0.0,
) -> dict[str, Any]:
    """Takes aggregated metrics and computes the final metrics for a run (e.g. one method).
    Args:
        confmat (np.ndarray): Confusion matrix for the method.
        config (dict): Configuration
        info (dict): Information about the method, such as patch size, stride, margin, padding, stitching method.
        method_times (dict): Dictionary containing the timings for the run, one element corresponding to one whole image
        method_patches (list[int]): List of number of patches for the run, one element corresponding to the number of patches for one whole image
        area (float): Total area processed in square meters. Default is 0.0.
        carbon (float): Total carbon emissions in kg. Default is 0.0.
    """

    patch_size = info["patch_size"]
    stride = info["stride"]
    margin = info["margin"]
    padding = info["padding"]
    stitching = info["stitching"]

    classes = config["classes"]
    n_classes = len(classes)

    total_patches = int(np.sum(method_patches))
    mean_patches = np.mean(method_patches) if len(method_patches) > 0 else 0

    # compute metrics for the group
    norm_confmat = confmat / total_patches if total_patches > 0 else confmat
    confmat_cleaned = clean_confmat(norm_confmat, config)

    # metrics
    with np.errstate(divide="ignore", invalid="ignore"):
        # nans are handled dont worry
        per_c_ious, avg_ious, std_ious = class_IoU(confmat_cleaned)
        ovr_acc = overall_accuracy(confmat_cleaned)
        per_c_fscore, avg_fscore, std_fscore = class_fscore(confmat_cleaned)

        # get timings

        # dict of str : list of float
        norm_data_prep = np.array(method_times["data_prep_time"]) / method_patches
        avg_data_prep = np.mean(norm_data_prep) * mean_patches
        sigma_data_prep = np.std(norm_data_prep)  # std dev of time normalized per patch

        norm_inference = np.array(method_times["pure_infer_time"]) / method_patches
        avg_inference = np.mean(norm_inference) * mean_patches
        sigma_inference = np.std(norm_inference)

        norm_write = np.array(method_times["data_write_time"]) / method_patches
        avg_write = np.mean(norm_write) * mean_patches
        sigma_write = np.std(norm_write)

        norm_time = np.array(method_times["total_time"]) / method_patches
        avg_time = np.mean(norm_time) * mean_patches
        sigma_time = np.std(norm_time)

    metrics = {
        "Method parameters": [
            "model name",
            "patch size",
            "stride",
            "margin",
            "padding",
            "stitching method",
        ],
        "Parameters values": [
            config["model_name"],
            patch_size,
            stride,
            margin,
            padding,
            stitching,
        ],
        "Avg_metrics_name": [
            "mIoU",
            "Overall Accuracy",
            "Fscore",
            "Data preparation time in ms",
            "Inference time in ms",
            "Data writing time in ms",
            "Total time in ms",
            "Total patches processed",
            "Total area processed in m2",
            "Total carbon emissions in kg",
        ],
        "Avg_metrics": [
            avg_ious,
            ovr_acc,
            avg_fscore,
            avg_data_prep,
            avg_inference,
            avg_write,
            avg_time,
            total_patches,
            area,
            carbon,
        ],
        "std_dev_metrics": [
            std_ious,
            "undefined",
            std_fscore,
            sigma_data_prep,
            sigma_inference,
            sigma_write,
            sigma_time,
            "undefined",
            "undefined",
            "undefined",
        ],
        "classes": [classes[i][1] for i in range(1, n_classes + 1)],
        "per_class_iou": list(per_c_ious),
        "per_class_fscore": list(per_c_fscore),
    }
    return metrics


def batch_metrics(config: Config, truth_dir: Path) -> list[dict[str, Any]]:
    """Compute metrics for each method in the batch mode. The metrics are computed for the whole image, not per patch. Computation is based on the image files.
    Args:
        config (dict): Configuration, in which the parameters for the inference are specified
        truth_dir (Path): Path to the ground truth directory.
    Returns:
        metrics_file (list): List of dictionaries containing the metrics for each method. Temporal metrics correspond to the estimated processing time of one full image for that method, computed with an average weighted by number of patches per image.
    """

    metrics_file = []
    df = collect_paths_truth(config, truth_dir)
    n_classes = len(config["classes"])

    grouped = df.groupby("method")

    # metrics for each method
    print("Computing metrics...")
    for method, group in tqdm(grouped, desc="Computing metrics...", total=len(grouped)):

        # method parameters
        info = extract_method(str(method))

        pred_paths = group["pred_path"].tolist()
        gt_paths = group["truth_path"].tolist()

        sum_confmat = np.zeros((n_classes, n_classes))

        for pred_path, truth_path in zip(pred_paths, gt_paths):

            sum_confmat = add_confusion(
                Path(pred_path),
                Path(truth_path),
                sum_confmat,
                n_classes,
                info["stride"],
            )

        method_times = config.get("times", {}).get(method, [])
        method_patches = config.get("nb_patches", {}).get(method, [])

        metrics = process_metrics(
            sum_confmat, config, info, method_times, method_patches
        )

        metrics_file.append(metrics)

    return metrics_file


def error_rate_loop(
    truth_dir: Path, out_dir: Path, pred_dir: Path, autoscale=True
) -> None:
    """Args:
    pred_dir (Path): the directory with predictions
    out_dir (Path): the output directory for the error rate
    truth_dir (Path): the ground truth directory of the department"""
    dic = {}

    # get all tif files in the pred_dir
    tif_files = list(pred_dir.rglob("*.tif"))

    for pred_path in tqdm(tif_files, desc="Computing error rate"):

        # get corresponding truth file
        truth_file = get_truth_path(pred_path, truth_dir)

        dic = error_rate_patch(
            truth_file=truth_file,
            out_dir=out_dir,
            pred_path=pred_path,
            dic=dic,
            save=False,
            autoscale=autoscale,
        )

    # aggregate the error rate for each method over all keys
    methods = dict()
    total = dict()
    for key in dic.keys():
        # get the method name
        method = info_extract(Path(key))["method"]
        if method not in methods.keys():
            methods[method] = dic[key]
            total[method] = 1
        else:
            methods[method] += dic[key]
            total[method] += 1

    for key in methods.keys():
        methods[key] = methods[key] / total[key]
        # save the error rate as a png
        if autoscale:
            vmin = np.min(methods[key])
            vmax = np.max(methods[key])
        else:
            vmin = 0.025
            vmax = 0.25
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(
            methods[key], cmap="plasma", interpolation="nearest", vmin=vmin, vmax=vmax
        )
        plt.colorbar()
        plt.title("Error Rate for method : \n" + key)
        plt.savefig(str(out_dir / f"error_rate_{key}.png"))
        plt.close()

        print(f"Error rate saved to {out_dir / f'error_rate_{key}.png'}")

        # Error vs distance to edge

        h, w = methods[key].shape
        yy, xx = np.indices((h, w))
        dist_edge = np.minimum.reduce([yy, h - 1 - yy, xx, w - 1 - xx])
        max_dist = dist_edge.max()
        error_vs_dist = [
            methods[key][dist_edge == d].mean() for d in range(max_dist + 1)
        ]

        plt.figure(figsize=(8, 6))
        plt.plot(range(max_dist + 1), error_vs_dist, marker="o")
        plt.xlabel("Distance to Edge (pixels)")
        plt.ylabel("Average Error Rate")
        plt.title("Error vs Distance-to-Edge : \n" + key)
        plt.grid(True)
        plt.savefig(str(out_dir / f"error_vs_edge_{key}.png"))
        plt.close()


# not incorporated in the pipeline, but maybe as option ?
def error_rate_patch(
    truth_file: Path,
    out_dir: Path,
    pred_path: Path,
    dic: dict,
    save: bool,
    autoscale: bool = True,
) -> dict[str, np.ndarray]:
    """Compute the error rate per patch for a single prediction-label pair.
    Full paths are needed for zone extraction.
    """

    # slice prediction parameters

    file_info = info_extract(pred_path)
    dpt, zone = file_info["dpt"], file_info["zone"]
    patch_size, stride, margin = (
        file_info["patch_size"],
        file_info["stride"],
        file_info["margin"],
    )

    full_method = str(pred_path).split("/")[-1].split("-ARGMAX-S_")[1].split(".tif")[0]

    region_check = f"{dpt}/{zone}/himom.tif"

    # sanity check
    assert out_dir is not None, "Please provide an output path for the metrics"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    # for testing purposes we comment out
    truth_path = valid_truth({"truth_path": truth_file, "input_img_path": region_check})
    # truth_path = truth_file

    # load images in arrays
    # PIL struggles (multi band, 16 bit, float32)
    with rasterio.open(truth_path) as src:
        target = src.read(1) - 1  # to match the prediction
    with rasterio.open(pred_path) as src:
        pred = src.read(1)

    # slice into patches
    img_size = target.shape[0], target.shape[1]
    patches = slice_pixels(
        img_size,
        patch_size,
        margin,
        stride,
    )
    # set margin to 0 for relevance
    effective_patch_size = patch_size - 2 * margin
    out_array = np.zeros(
        (effective_patch_size, effective_patch_size),
    )
    # iterate over the patches and access images using patches indices ?
    for patch in patches:
        bottom, top, left, right = patch

        target_patch = target[bottom:top, left:right]
        pred_patch = pred[bottom:top, left:right]

        # compute the error rate : for each pixel, increment if different
        # hard error rate
        # replace / evaluate soft confidence
        out_array += np.where(target_patch != pred_patch, 1, 0)

    out_array = out_array / len(patches)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = out_path / f"error_rate_{full_method}_{datetime.datetime.now()}.png"

    # better visualization
    from scipy.ndimage import gaussian_filter

    out_array = gaussian_filter(out_array, sigma=2)

    if autoscale:
        vmin = np.min(out_array)
        vmax = np.max(out_array)
    else:
        vmin = 0.025
        vmax = 0.25

    if save:
        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(
            out_array, cmap="plasma", interpolation="nearest", vmin=vmin, vmax=vmax  # type: ignore
        )
        plt.colorbar()
        plt.title("Error Rate for method : \n" + full_method)
        plt.savefig(str(out_path))
        plt.close()
        print(f"Error rate saved to {out_path}")

    dic[pred_path] = out_array

    return dic


#### ANALYSIS ####
def load_metrics_json(json_path: Path) -> list[dict]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def flatten_as_dict(
    metrics_dict: list[dict], sort_per_param: str
) -> list[dict[str, Any]]:
    """
    Flatten the metrics dictionary into a list of dictionaries.
    Each dictionary contains the method parameters and the metrics.
    Args:
        metrics_dict (list[dict]): List of dictionaries containing the metrics for each method.
        sort_per_param (str): Parameter to sort the flattened metrics by. If empty, no sorting is applied.
    Returns:
        flattened (list[dict[str, Any]]): Flattened metrics.
    """

    flattened = []

    for method in metrics_dict:
        flat = {}

        # Flatten parameters
        for k, v in zip(method["Method parameters"], method["Parameters values"]):
            flat[f"param/{k}"] = v

        # Flatten avg metrics
        for k, v in zip(method["Avg_metrics_name"], method["Avg_metrics"]):
            flat[f"avg/{k}"] = v

        # Flatten class-level IoU and Fscore
        classes = method["classes"]
        for cls, iou in zip(classes, method["per_class_iou"]):
            flat[f"class/{cls}/iou"] = iou
        for cls, fscore in zip(classes, method["per_class_fscore"]):
            flat[f"class/{cls}/fscore"] = fscore

        flattened.append(flat)

    # Sort by a specific parameter if provided
    if sort_per_param:
        # Allow for secondary sorting by accepting a tuple, e.g., "patch size,stride"
        params = [p.strip() for p in sort_per_param.split(",")]
        if len(params) == 1:
            flattened.sort(key=lambda x: x.get(f"param/{params[0]}", 0))
        else:
            flattened.sort(key=lambda x: tuple(x.get(f"param/{p}", 0) for p in params))

    return flattened


def log_to_WB(metrics_path: Path, sort_per_param: str = "") -> None:
    """Log the metrics to Weights & Biases (W&B) for visualization and tracking.
    Args:
        metrics_path (Path): Path to the metrics JSON file.
        sort_per_param (str): Parameter to sort the flattened metrics by. If empty, no sorting is applied.
            -  model name, patch size, stride, margin, padding, stitching method
    """

    name = metrics_path.parent.name

    metrics_dict = load_metrics_json(metrics_path)
    flattened = flatten_as_dict(metrics_dict, sort_per_param)

    # config = {}
    # for parameters that do not change, , like
    # model name, model type (onnx, pt), device
    # decomposition of path into config
    name_elements = name.split("_")[-3:]
    config = {
        "model_name": name_elements[0],
        "model_type": name_elements[1],
        "device": name_elements[2],
    }

    wandb.init(project="semantic-segmentation-eval", config=config, name=name)
    # Log to W&B (each method = one step)

    for flat in flattened:
        wandb.log(flat)

    wandb.finish()


def generate_readable_colors(n):
    """
    Generate n visually distinct, readable colors using HSV -> RGB.
    Saturation and Value are controlled to avoid very light colors.
    """
    colors = []
    for i in range(n):
        h = i / n  # spread hues evenly
        s = 0.7  # high saturation
        v = 0.7  # medium brightness
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((r, g, b))
    return colors


def compute_colors(values, baseline_val, min_intensity=0.3, max_intensity=0.9):
    """
    Diverging pastel-to-dark colors:
    - Close to baseline → pastel (light)
    - Far from baseline → dark
    - Baseline → ochre
    """

    min_light = 0.5
    max_light = 0.9
    min_sat = 0.3
    max_sat = 1.0

    values = np.array(values)
    distances = np.abs(values - baseline_val)
    max_dist = distances.max() + 1e-12  # avoid div by zero

    colors = []
    for v, d in zip(values, distances):
        if np.isclose(v, baseline_val):
            colors.append("sandybrown")  # baseline ochre
        else:
            # Scale lightness and saturation with distance
            lightness = max_light - (d / max_dist) * (max_light - min_light)
            saturation = max_sat - (d / max_dist) * (max_sat - min_sat)
            if v < baseline_val:
                h = 1 / 3  # green
            else:
                h = 0  # red
            r, g, b = colorsys.hls_to_rgb(h, lightness, saturation)
            colors.append((r, g, b))
    return colors


def plot_experiments(json_files, baseline=None, out_path="plot.png"):
    miou_vals = []
    time_vals = []
    labels = []

    # Load metrics
    for f in json_files:
        with open(f, "r") as jf:
            data = json.load(jf)[0]
        miou_idx = data["Avg_metrics_name"].index("mIoU")
        inf_idx = data["Avg_metrics_name"].index("Inference time in ms")

        miou_vals.append(data["Avg_metrics"][miou_idx])
        time_vals.append(data["Avg_metrics"][inf_idx])
        labels.append(os.path.basename(os.path.dirname(f)))

    n_points = len(json_files)

    # Identify baseline
    baseline_idx = None
    if baseline is not None:

        for i, f in enumerate(json_files):
            if os.path.samefile(f, baseline):
                baseline_idx = i
                break
        else:
            raise ValueError(f"Baseline file {baseline} not found in json_files list")

    plt.figure(figsize=(8, 8))

    # Generate readable colors for experiments (exclude baseline)
    colors = generate_readable_colors(n_points)
    if baseline_idx is not None:
        colors[baseline_idx] = (1.0, 0.0, 0.0)  # red for baseline (cross)

        x_base = time_vals[baseline_idx]
        y_base = miou_vals[baseline_idx]
        # Draw light gray cross lines

        plt.axvline(x=x_base, color="gray", linestyle="--", linewidth=1, zorder=1)
        plt.axhline(y=y_base, color="gray", linestyle="--", linewidth=1, zorder=1)

    texts = []

    # Precompute offsets in a radial pattern
    radius = 0.02  # fraction of axis range
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    dxs = radius * np.cos(angles) * (max(time_vals) - min(time_vals))
    dys = radius * np.sin(angles) * (max(miou_vals) - min(miou_vals))

    for i, (x, y, label) in enumerate(zip(time_vals, miou_vals, labels)):
        if i == baseline_idx:
            plt.scatter(
                x, y, color="red", marker="x", s=150, linewidth=3, label="Baseline"
            )
        else:
            plt.scatter(x, y, color=colors[i], s=80, label="_nolegend_")
            texts.append(
                plt.text(x + dxs[i], y + dys[i], label, fontsize=12, color=colors[i])
            )

    plt.xlabel("Inference Time (ms)")
    plt.ylabel("mIoU")
    plt.title("mIoU vs Inference Time")

    adjust_text(
        texts, only_move={"points": "none", "texts": "xy"}, expand_text=(1.2, 1.2)
    )

    # Legend: show only baseline and "experiment" marker
    plt.scatter([], [], color="blue", s=80, label="Experiment")  # representative dot
    plt.legend()

    # Scale axes with small margin
    all_x = time_vals
    all_y = miou_vals
    plt.xlim(
        min(all_x) - 0.05 * (max(all_x) - min(all_x)),
        max(all_x) + 0.05 * (max(all_x) - min(all_x)),
    )
    plt.ylim(
        min(all_y) - 0.05 * (max(all_y) - min(all_y)),
        max(all_y) + 0.05 * (max(all_y) - min(all_y)),
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_carbon_emissions(json_files, baseline=None, out_path="carbon_plot.png"):
    names = []
    values = []

    # Load carbon emission from JSONs
    for f in json_files:
        with open(f, "r") as jf:
            data = json.load(jf)[0]
        # Find indices
        carbon_idx = data["Avg_metrics_name"].index("Total carbon emissions in kg")
        area_idx = data["Avg_metrics_name"].index("Total area processed in m2")
        carbon_kg = data["Avg_metrics"][carbon_idx]
        area_m2 = data["Avg_metrics"][area_idx]

        # Convert to grams per km²
        carbon_g_per_km2 = carbon_kg * 1e9 / area_m2
        values.append(carbon_g_per_km2)
        names.append(os.path.basename(os.path.dirname(f)))

    # Separate baseline if provided
    baseline_val = None
    baseline_name = ""
    other_names = []
    other_values = []
    for n, v, f in zip(names, values, json_files):
        if baseline is not None and os.path.samefile(f, baseline):
            baseline_val = v
            baseline_name = n
        else:
            other_names.append(n)
            other_values.append(v)

    # Sort other experiments ascending
    sorted_items = sorted(zip(other_names, other_values), key=lambda x: x[1])

    if sorted_items:
        sorted_names, sorted_values = zip(*sorted_items)
        sorted_names = list(sorted_names)  # convert tuple to list
        sorted_values = list(sorted_values)  # convert tuple to list
    else:
        sorted_names, sorted_values = [], []

    # Combine baseline first
    if baseline_val is not None:
        final_names = [baseline_name] + sorted_names
        final_values = [baseline_val] + sorted_values
        colors = ["red"] + generate_readable_colors(len(sorted_names))
    else:
        final_names = sorted_names
        final_values = sorted_values
        colors = generate_readable_colors(len(sorted_names))

    # Plot
    plt.figure(figsize=(12, 6))
    bars = plt.bar(
        final_names,
        final_values,
        color=compute_colors(final_values, baseline_val),
        width=0.6,
    )

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01 * max(final_values),
            f"{height:.6f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    padding_ratio = 0.10  # 5% of range
    y_max = max(final_values)
    y_range = y_max
    plt.ylim(0, y_max + padding_ratio * y_range)

    plt.ylabel("Carbon Emissions (g / km²)")
    plt.title("Carbon Emissions Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


#### RAM USAGE ####
def get_ram_usage_mb():
    """Get the current RAM usage of the process in MB.
    Measuring host RAM and not GPU memory"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # RAM usage in MB


def model_ram_compare(ckpt_list: list[str]) -> None:
    """
    Compare the RAM usage of different models loaded from their checkpoints.
    """
    ram_usage = {}

    model_config = {
        "model_weights": "",
        "model_framework": {
            "SegmentationModelsPytorch": {"encoder_decoder": "resnet34_unet"},
            "model_provider": "",  # or "SegmentationModelsPytorch"
            "HuggingFace": {"org_model": "openmmlab/upernet-swin-small"},
        },
        "channels": [1, 2, 3],
        "n_classes": 19,
    }

    for ckpt in ckpt_list:

        print(f"Loading model from {ckpt}...")
        gc.collect()

        model_config["model_weights"] = ckpt

        resnet = Path(ckpt).stem
        resnet = "resnet" in resnet

        if resnet:
            model_config["model_framework"][
                "model_provider"
            ] = "SegmentationModelsPytorch"
        else:
            model_config["model_framework"]["model_provider"] = "HuggingFace"

        before_ram = get_ram_usage_mb()

        model = load_model(model_config)

        after_ram = get_ram_usage_mb()
        ram_usage[ckpt] = after_ram - before_ram

        del model
        gc.collect()

    """# adding onnx model "loading"
    onnx_ckpt = "/media/DATA/INFERENCE_HS/MODELS_IA/FLAIR1/swin-upernet-small_IRV_SET1/checkpoints/openmmlab/upernet-swin-small_cpu_1x3x512x512.onnx"

    gc.collect()
    before_ram = get_ram_usage_mb()

    ort_session = ort.InferenceSession(onnx_ckpt, providers=["CPUExecutionProvider"])

    after_ram = get_ram_usage_mb()
    ram_usage[onnx_ckpt] = after_ram - before_ram

    del ort_session
    gc.collect()"""

    print("\nRAM Usage Comparison:")
    for ckpt, usage in ram_usage.items():
        print(f"{Path(ckpt).stem}: {usage:.2f} MB")


def error_rate_loop_to_center(
    truth_dir: Path, out_dir: Path, pred_dir: Path, autoscale=True
) -> None:
    """Aggregate error maps and produce relative heatmap + relative error-vs-edge plots."""
    dic = {}
    tif_files = list(pred_dir.rglob("*.tif"))

    for pred_path in tqdm(tif_files, desc="Computing error rate"):
        truth_file = get_truth_path(pred_path, truth_dir)
        dic = error_rate_patch(
            truth_file=truth_file,
            out_dir=out_dir,
            pred_path=pred_path,
            dic=dic,
            save=False,
            autoscale=autoscale,
        )

    methods = {}
    total = {}
    for key in dic.keys():
        method = info_extract(Path(key))["method"]
        if method not in methods:
            methods[method] = dic[key]
            total[method] = 1
        else:
            methods[method] += dic[key]
            total[method] += 1

    for key in methods.keys():
        methods[key] = methods[key] / total[key]

        # === Normalize by center error ===
        h, w = methods[key].shape
        center_error = methods[key][h // 2, w // 2]
        if center_error == 0:
            center_error = 1e-8
        rel_map = methods[key] / center_error

        # === Heatmap (relative to center) ===
        if autoscale:
            vmin, vmax = np.min(rel_map), np.max(rel_map)
        else:
            vmin, vmax = 0.5, 2.0  # around 1.0 baseline

        plt.figure(figsize=(10, 10))
        plt.axis("off")
        plt.imshow(
            rel_map, cmap="plasma", interpolation="nearest", vmin=vmin, vmax=vmax
        )
        plt.colorbar()
        plt.title("Relative Error Map (center = 1.0) : \n" + key)
        plt.savefig(str(out_dir / f"error_rate_relative_{key}.png"))
        plt.close()

        # === Error vs distance-to-edge (relative) ===
        yy, xx = np.indices((h, w))
        dist_edge = np.minimum.reduce([yy, h - 1 - yy, xx, w - 1 - xx])
        max_dist = dist_edge.max()
        error_vs_dist = [
            methods[key][dist_edge == d].mean() for d in range(max_dist + 1)
        ]

        error_vs_dist = [val / center_error for val in error_vs_dist]

        plt.figure(figsize=(8, 6))
        plt.plot(range(max_dist + 1), error_vs_dist, marker="o", label="Relative error")

        # find distance(s) where error == 1.0 (± tolerance)
        tol = 1e-2
        equal_idx = [d for d, val in enumerate(error_vs_dist) if abs(val - 1.0) < tol]
        if equal_idx:
            d0 = equal_idx[0]
            plt.scatter(
                d0, error_vs_dist[d0], color="red", zorder=5, label=f"=1 at d={d0}"
            )
            plt.annotate(
                f"d={d0}",
                (d0, error_vs_dist[d0]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                color="red",
            )

        plt.xlabel("Distance to Edge (pixels)")
        plt.ylabel("Relative Error Rate (center = 1.0)")
        plt.title("Relative Error vs Distance-to-Edge : \n" + key)
        plt.grid(True)
        plt.legend()
        plt.savefig(str(out_dir / f"error_vs_edge_relative_{key}.png"))
        plt.close()


if __name__ == "__main__":

    # extract metrics paths from the output directory
    dir = "/var/tmp/shys/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/swin-upernet-small/D037_2021/out20250829"

    metrics_paths = list(str(p) for p in Path(dir).rglob("**/metrics.json"))

    # remove sweep metrics
    metrics_paths = [p for p in metrics_paths if ("sweep" not in str(p))]

    baseline = "/var/tmp/shys/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/swin-upernet-small/D037_2021/out20250821/CPU/E0_baseline/metrics.json"

    # plot_experiments(metrics_paths, baseline=baseline, out_path="cpu_plot.png")

    path = "/var/tmp/shys/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/swin-upernet-small/D037_2021/out20250828/20250828151342_E2-2_margin-sweep_patch=1024/metrics.json"
    # log_to_WB(Path(path), sort_per_param="patch size, margin")

    config = "/home/SHys/FLAIR-1/configs/expe/E0_baseline_default.yaml"

    truth_root = "/var/tmp/shys/INFERENCE_HS/DATA/dataset_zone_last/labels_raster/FLAIR_19/D037_2021"

    error_rate_loop_to_center(
        truth_dir=Path(truth_root),
        out_dir=Path(dir) / "error_rate_swin_center_margin",
        pred_dir=Path(dir) / "small_pytorch_gpu",
    )
