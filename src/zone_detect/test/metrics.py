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

from src.zone_detect.test.tiles import get_stride
from src.zone_detect.test.pixel_operation import slice_pixels


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

        # extract region info
        region_id = str(pred_files[0].name).split("_IRC-ARGMAX-S_")[0]
        dpt, zone_name = region_id.split("_")[:2], region_id.split("_")[2:]
        zone_name = "_".join(zone_name)

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
        # nans are handled dont worry
        per_c_ious, avg_ious = class_IoU(confmat_cleaned)
        ovr_acc = overall_accuracy(confmat_cleaned)
        per_c_fscore, avg_fscore = class_fscore(confmat_cleaned)

    # save metrics to a json file : raw or post-stitching
    out = Path(out_json).with_suffix(".json")
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
        # add a new line for each entry


def batch_metrics(config: dict, gt_dir: str) -> list:
    metrics_file = []
    df = collect_paths_truth(config, gt_dir)

    grouped = df.groupby("method")

    # metrics for each method
    # TODO : add a progress bar
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
        tile_size, stride, margin, padding, stitching = str(method).split("_")
        tile_size = int(tile_size.split("=")[1])
        stride = int(stride.split("=")[1])
        margin = int(margin.split("=")[1])
        padding = padding.split("=")[1]
        stitching = stitching.split("=")[1]

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
                tile_size,
                stride,
                margin,
                padding,
                stitching,
            ],
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
        metrics_file.append(metrics)

    return metrics_file


# not incorporated in the pipeline, but maybe as option ?
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

    method = str(pred_path).split("IRC-ARGMAX-S_")[-1].split(".tif")[0]

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
