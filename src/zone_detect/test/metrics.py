import datetime
import json
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import rasterio
from rasterio.windows import Window
from sklearn.metrics import confusion_matrix
from src.zone_detect.main import run_from_config
from src.zone_detect.test.pixel_operation import slice_pixels


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


def valid_truth(config: dict) -> Path:
    truth_path = config["truth_path"]
    # verify coherence with input path
    sanity_check = config["input_img_path"].split("/")[-3:-1]  # zone
    if truth_path.split("/")[-3:-1] != sanity_check:
        raise ValueError(
            f"Ground truth path {truth_path} does not match input path {config['input_img_path']}"
        )
    return Path(truth_path)


# __________PIPELINE__________#
def batch_metrics_pipeline(
    inputs_dir: str, gt_dir: str, config: dict, out_json: str
) -> None:
    """
    Compute metrics for a batch of images.
    Args:
        inputs_dir (str): Path to the input directory containing images.
        gt_dir (str): Path to the ground truth directory.
        config (dict): Configuration, in whichthe parameters for the inference are specified
        out_csv (str): Path to the output CSV file for metrics. If the file exists, it will be overwritten.
    """

    # output file
    assert out_json is not None, "Please provide an output path for the metrics"

    # __________INFERENCE__________#
    inputs_folder = Path(inputs_dir)

    for zone in sorted(inputs_folder.iterdir()):

        # find an input file image
        if not zone.is_dir():
            continue
        irc_path = next(zone.glob("*IRC.tif"), None)
        if irc_path is None:
            continue

        config["input_img_path"] = str(irc_path)

        # Inference and saving the predictions
        run_from_config(config)

    # we have all the predictions in the output folder

    out = Path(out_json).with_suffix(".json")
    metrics_file = []
    path_collection = []
    gt_folder = Path(gt_dir)

    # get predictions
    pred_folder = Path(config["output_path"].split("/")[:-2])
    # after inference, output path is pred_folder/zone_name/timestamp

    # dataframe with pred path, gt path and method single string
    for zone in sorted(pred_folder.iterdir()):
        # find an input file image
        if not zone.is_dir():
            continue
        zone_name = zone.name

        zone_path = pred_folder / zone_name
        pred_files = list(zone_path.rglob("*.tif"))

        # corresponding ground truth
        gt_subfolder = gt_folder / zone_name
        gt_path = next(gt_subfolder.glob("*.tif"), None)

        for pred_path in pred_files:
            method_id = Path(pred_path.name.split("ARGMAX-IRC-S_")[1]).stem
            # zone_name_ARGMAX-IRC-S_size=128_stride=96_margin=32_padding=some-padding_stitching=exact_clipping
            path_collection.append(
                {
                    "pred_path": str(pred_path),
                    "gt_path": str(gt_path),
                    "method": method_id,
                }
            )

    df = pd.DataFrame(path_collection)
    grouped = df.groupby("method")

    # metrics for each method
    for method, group in grouped:

        pred_paths = group["pred_path"].tolist()
        gt_paths = group["gt_path"].tolist()

        patch_confusion_matrices = []
        for pred_path, gt_path in zip(pred_paths, gt_paths):
            try:
                target = (
                    np.array(Image.open(gt_path)) - 1
                )  # depending on classes starting at 1 or 0
                preds = np.array(Image.open(pred_path))
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

        #### CLEAN REGARDING WEIGHTS FOR METRICS CALC :
        # commented out, put back if needed
        weights = np.array([config["classes"][i][0] for i in config["classes"]])
        unused_classes = np.where(weights == 0)[0]
        confmat_cleaned = np.delete(sum_confmat, unused_classes, axis=0)  # remove rows
        confmat_cleaned = np.delete(confmat_cleaned, unused_classes, axis=1)

        # metrics
        per_c_ious, avg_ious = class_IoU(sum_confmat)
        ovr_acc = overall_accuracy(sum_confmat)
        per_c_fscore, avg_fscore = class_fscore(sum_confmat)

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

    # save the metrics to a json file
    json.dump(
        metrics_file,
        open(out, "w"),
    )
    print(f"Metrics saved to {out}")


def compute_metrics_patch(
    pred_patch: np.ndarray, window: Window, config: dict, out_json: str
) -> None:
    """
    Patch metrics can be computed before the stitching ,
    or once the whole image is built.
    Args:
        pred_patch (np.ndarray): Predicted patch.
        window (Window): Window object for the patch.
        config (dict): Configuration, in whichthe parameters for the inference are specified
        out_json (str): Path to the output JSON file for metrics. If the file exists, it will be overwritten.
            You better put in the name if it's raw (before stitching) or after.
    """

    # get ground truth path from config
    truth_path = valid_truth(config)

    with rasterio.open(truth_path) as src:
        # only supports argmax for now
        target = src.read(1, window=window)

    #### compute metrics
    # confusion matrix
    confmat = confusion_matrix(
        target.flatten(),
        pred_patch.flatten(),
        labels=list(range(int(len(config["classes"])))) + [255],
    )
    #### CLEAN REGARDING WEIGHTS FOR METRICS CALC :
    weights = np.array([config["classes"][i][0] for i in config["classes"]])
    unused_classes = np.where(weights == 0)[0]
    confmat_cleaned = np.delete(confmat, unused_classes, axis=0)  # remove rows
    confmat_cleaned = np.delete(
        confmat_cleaned, unused_classes, axis=1
    )  # remove columns

    per_c_ious, avg_ious = class_IoU(confmat_cleaned)
    ovr_acc = overall_accuracy(confmat_cleaned)
    per_c_fscore, avg_fscore = class_fscore(confmat_cleaned)

    # save metrics to a json file : raw or post-stitching
    out = Path(out_json).with_suffix(".json")
    key = (window.col_off, window.row_off)
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
    print(f"Metrics saved to {out}")


def error_rate_patch(config: dict, out_path: str, pred_filename: str = "") -> None:
    # output file
    assert out_path is not None, "Please provide an output path for the metrics"

    truth_path = valid_truth(config)
    out_array = np.zeros(
        (config["img_pixels_detection"], config["img_pixels_detection"])
    )

    # slice into patches
    patches = slice_pixels(
        config["img_pixels_detection"],
        config["img_pixels_detection"],
        config["margin"],
        config["stride"],
    )

    # prediction path
    pred_folder = Path(config["output_path"])
    pred_path = pred_folder / pred_filename
    if pred_filename == "":
        pred_path = next(pred_folder.glob("*.tif"), None)  # select the first one
    if pred_path is None:
        raise ValueError(f"No prediction file found in {pred_folder}")

    # load images in arrays
    target = np.array(Image.open(truth_path))
    pred = np.array(
        Image.open(pred_path).convert("L")
    )  # convert to grayscale because we have 2 bands

    # iterate over the patches and access images using patches indices ?
    for patch in patches:
        left, bottom, right, top = patch

        target_patch = target[bottom:top, left:right]
        pred_patch = pred[bottom:top, left:right]

        # compute the error rate
        # for each pixel, increment if ont equal
        out_array += np.where(target_patch != pred_patch, 1, 0)

    out_array = out_array / len(patches)

    # i'm afraid of the memory

    # useful info : method used, error rate

    method = pred_filename.split("ARGMAX-IRC-S_")[1]

    plt.imshow(out_array, cmap="hot", interpolation="nearest")
    plt.colorbar()
    plt.title("Error Rate for method : " + method)
    # plt.savefig( out_path + "/error_rate_" + method + "_" + str(datetime.datetime.now()) + ".png")
    plt.show()
