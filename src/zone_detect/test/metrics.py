import datetime
import json
from pathlib import Path
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from src.zone_detect.test.pixel_operation import slice_pixels
from src.zone_detect.utils import info_extract


#### UTILS ####
def clean_confmat(confmat: np.ndarray, config: dict) -> np.ndarray:
    #### CLEAN REGARDING WEIGHTS FOR METRICS CALC :
    weights = np.array([config["classes"][i][0] for i in config["classes"]])
    unused_classes = np.where(weights == 0)[0]
    confmat_cleaned = np.delete(confmat, unused_classes, axis=0)  # remove rows
    confmat_cleaned = np.delete(
        confmat_cleaned, unused_classes, axis=1
    )  # remove columns

    return confmat_cleaned


def valid_truth(config: dict) -> Path:
    """Check if the ground truth path is valid and coherent with the input path :
    the zone should be the same in both paths.
    """
    truth_path = config["truth_path"]
    # verify coherence with input path
    sanity_check = config["input_img_path"].split("/")[-3:-1]  # zone
    if truth_path.split("/")[-3:-1] != sanity_check:
        raise ValueError(
            f"Ground truth path {truth_path} does not match input path {config['input_img_path']}"
        )
    return Path(truth_path)


def collect_paths_truth(config: dict, gt_dir: str) -> pd.DataFrame:
    path_collection = []
    gt_folder = Path(gt_dir)

    # get predictions
    pred_folder = Path(config["output_path"])
    timed_folders = [p for p in pred_folder.iterdir() if p.is_dir()]

    # dataframe with pred path, gt path and method single string
    for timestamp in sorted(timed_folders):
        pred_files = list(timestamp.rglob("*.tif"))

        zone_name = info_extract(str(pred_files[0]))["zone"]

        # corresponding ground truth
        # we consider gt_folder the dpt folder
        gt_subfolder = gt_folder / zone_name
        gt_path = next(gt_subfolder.glob("*.tif"), None)

        for pred_path in pred_files:
            method_id = Path(pred_path.name.split("IRC-ARGMAX-S_")[1]).stem
            # zone_name_IRC-ARGMAX-S_size=128_stride=96_margin=32_padding=some-padding_stitching=exact_clipping
            path_collection.append(
                {
                    "pred_path": str(pred_path),
                    "gt_path": str(gt_path),
                    "method": method_id,
                }
            )
    return pd.DataFrame(path_collection)


#### METRICS ####
def overall_accuracy(npcm):
    oa = np.trace(npcm) / npcm.sum()
    return 100 * oa


def class_IoU(npcm):
    ious = (
        100
        * np.diag(npcm)
        / (np.sum(npcm, axis=1) + np.sum(npcm, axis=0) - np.diag(npcm))
    )
    ious[np.isnan(ious)] = 0
    return ious, np.mean(ious)


def class_precision(npcm):
    precision = 100 * np.diag(npcm) / np.sum(npcm, axis=0)
    precision[np.isnan(precision)] = 0
    return precision, np.mean(precision)


def class_recall(npcm):
    recall = 100 * np.diag(npcm) / np.sum(npcm, axis=1)
    recall[np.isnan(recall)] = 0
    return recall, np.mean(recall)


def class_fscore(npcm):
    precision = class_precision(npcm)[0]
    recall = class_recall(npcm)[0]
    fscore = 2 * (precision * recall) / (precision + recall)
    fscore[np.isnan(fscore)] = 0
    return fscore, np.mean(fscore)


#### COMPUTATION ####
def compute_metrics_patch(
    pred_patch: np.ndarray, window: Window, config: dict, method: str, out_json: str
) -> None:
    """
    Patch metrics can be computed before the stitching ,
    or once the whole image is built.
    Average etrics are not exactly relevant because of the classes absent from a patch.
    Args:
        pred_patch (np.ndarray): Predicted patch.
        window (Window): Window object for the patch.
        config (dict): Configuration, in which the parameters for the inference are specified
        out_json (str): Path to the output JSON file for metrics. If the file exists, it will be overwritten.
            You better put in the name if it's raw (before stitching) or after.
    """

    # get ground truth path from config
    truth_path = valid_truth(config)

    with rasterio.open(truth_path) as src:
        # only supports argmax for now
        target = src.read(1, window=window) - 1

    #### compute metrics
    # confusion matrix
    confmat = confusion_matrix(
        target.flatten(),
        pred_patch[0].flatten(),
        labels=list(range(int(len(config["classes"])))) + [255],
    )

    confmat_cleaned = clean_confmat(confmat, config)

    with np.errstate(divide="ignore", invalid="ignore"):
        # nans are handled don't worry
        per_c_ious, avg_ious = class_IoU(confmat_cleaned)
        ovr_acc = overall_accuracy(confmat_cleaned)
        per_c_fscore, avg_fscore = class_fscore(confmat_cleaned)

    # save metrics to a json file : raw or post-stitching
    out = Path(out_json)
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
            "classes": list(
                np.array([config["classes"][i][1] for i in config["classes"]])
            ),
            "per_class_iou": list(per_c_ious),
            "per_class_fscore": list(per_c_fscore),
        }
    }
    with open(out, "a") as f:
        json.dump(metrics, f)
        f.write("\n")


def batch_metrics(config: dict, gt_dir: str) -> list:
    metrics_file = []
    df = collect_paths_truth(config, gt_dir)

    grouped = df.groupby("method")

    # metrics for each method
    for method, group in tqdm(grouped, desc="Computing metrics"):

        pred_paths = group["pred_path"].tolist()
        gt_paths = group["gt_path"].tolist()

        patch_confusion_matrices = []
        for pred_path, gt_path in zip(pred_paths, gt_paths):
            try:
                # loading
                with rasterio.open(pred_path) as src:
                    preds = src.read(1)
                with rasterio.open(gt_path) as src:
                    target = src.read(1) - 1

                patch_confusion_matrices.append(
                    confusion_matrix(
                        target.flatten(),
                        preds.flatten(),
                        labels=list(range(int(len(config["classes"])))) + [255],
                    )
                )
            except Exception as e:
                print(f"Error processing {pred_path} and {gt_path}: {e}")

        # compute metrics for the group
        sum_confmat = np.sum(patch_confusion_matrices, axis=0)
        confmat_cleaned = clean_confmat(sum_confmat, config)

        # metrics
        with np.errstate(divide="ignore", invalid="ignore"):
            # nans are handled dont worry
            per_c_ious, avg_ious = class_IoU(confmat_cleaned)
            ovr_acc = overall_accuracy(confmat_cleaned)
            per_c_fscore, avg_fscore = class_fscore(confmat_cleaned)
            avg_time = np.mean(config["times"][method])

        # method parameters
        info = info_extract("/" + str(method) + ".tif")
        patch_size = info["patch_size"]
        stride = info["stride"]
        margin = info["margin"]
        padding = info["padding"]
        stitching = info["stitching"]

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
            "Avg_metrics_name": ["mIoU", "Overall Accuracy", "Fscore", "Time"],
            "Avg_metrics": [
                avg_ious,
                ovr_acc,
                avg_fscore,
                avg_time,
            ],
            "classes": list(
                np.array([config["classes"][i][1] for i in config["classes"]])
            ),
            "per_class_iou": list(per_c_ious),
            "per_class_fscore": list(per_c_fscore),
        }
        metrics_file.append(metrics)

    return metrics_file


# not incorporated in the pipeline, but maybe as option ?
def error_rate_patch(truth_file: str, out_dir: str, pred_file: str) -> None:
    """Compute the error rate per patch for a given input.
    You need to provide full paths
    """

    # slice prediction parameters
    pred_path = Path(pred_file)

    file_info = info_extract(pred_file)
    dpt, zone = file_info["dpt"], file_info["zone"]
    patch_size, stride, margin = (
        file_info["patch_size"],
        file_info["stride"],
        file_info["margin"],
    )

    full_method = pred_file.split("/")[-1].split("_IRC-ARGMAX-S_")[1].split(".tif")[0]

    region_check = f"{dpt}/{zone}/himom.tif"

    # sanity check
    assert out_dir is not None, "Please provide an output path for the metrics"
    truth_path = valid_truth({"truth_path": truth_file, "input_img_path": region_check})

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
        patch_size,
        margin,
        stride,
    )
    effective_patch_size = patch_size - 2 * margin
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

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = out_path / f"error_rate_{full_method}_{datetime.datetime.now()}.png"

    # save the error rate as a png
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(out_array, cmap="hot", interpolation="nearest", vmin=0.025, vmax=0.25)
    plt.colorbar()
    plt.title("Error Rate for method : \n" + full_method)
    plt.savefig(str(out_path))
    plt.close()
    print(f"Error rate saved to {out_path}")


#### ANALYSIS ####
def load_metrics_json(json_path: str) -> dict:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def flatten_metrics(metrics: dict) -> pd.DataFrame:
    """
    Flatten the metrics dictionary into a pandas DataFrame.
    """
    flat_metrics = []
    for key, value in metrics.items():
        flat_metrics.append(
            {
                "key": key,
                **{
                    k: v
                    for k, v in value.items()
                    if k not in ["Avg_metrics_name", "Avg_metrics"]
                },
            }
        )
    return pd.DataFrame(flat_metrics)


def analyze_param(df: pd.DataFrame, param: str, metric: str) -> pd.DataFrame:
    """
    Analyze the metrics for a given parameter.
    """
    # filter the dataframe for the given parameter
    df = df[df["key"].str.contains(param)]
    # extract the parameter values
    df[param] = df["key"].str.extract(rf"{param}=(\d+)")
    # convert to numeric
    df[param] = pd.to_numeric(df[param])
    # group by parameter and compute the mean of the metric
    df = df.groupby(param).agg({metric: "mean"}).reset_index()
    return df


def plot_metrics(df: pd.DataFrame, param: str, metric: str) -> None:
    """
    Plot the metrics for a given parameter.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df[param], df[metric], marker="o")
    plt.xlabel(param)
    plt.ylabel(metric)
    plt.title(f"{metric} vs {param}")
    plt.grid()
    plt.show()
    plt.savefig(f"{param}_{metric}.png")
    plt.close()
    print(f"Plot saved to {param}_{metric}.png")
