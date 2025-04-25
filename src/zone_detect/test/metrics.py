import datetime
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from src.zone_detect.main import run_from_config


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


def batch_metrics_pipeline(
    inputs_dir: str, gt_dir: str, config: dict, out_csv: str
) -> None:

    # get output file
    assert out_csv is not None, "Please provide an output path for the metrics"

    # get inputs
    inputs_folder = Path(inputs_dir)

    for x in sorted(inputs_folder.iterdir()):

        # find an input file image
        if not x.is_dir():
            continue
        irc_path = next(x.glob("*IRC.tif"), None)
        if irc_path is None:
            continue

        config["input_img_path"] = str(irc_path)

        # Inference and saving the predictions
        run_from_config(config)

    # we have all the predictions in the output folder

    out = Path(out_csv).with_suffix(".csv")
    # if file exists, overwrite
    metrics_file = {}

    gt_folder = Path(gt_dir)

    # get predictions
    pred_folder = Path(config["output_path"].split("/")[:-2])
    # after inference, output path is .../zone_name/timestamp

    # dataframe with pred path, gt path and method single string
    for x in sorted(pred_folder.iterdir()):
        # find an input file image
        if not x.is_dir():
            continue
        name = x.name

        pred_root = pred_folder / name
        pred_files = list(pred_root.rglob("*.tif"))

        # corresponding ground truth
        gt_subfolder = gt_folder / name
        gt_path = next(gt_subfolder.glob("*.tif"), None)

        method_id = [
            Path(file.name.split("ARGMAX-IRC-S_")[1]).stem for file in pred_files
        ]

    # _____________________________________________________#

    sum_confmat = np.sum(patch_confusion_matrices, axis=0)
    #### CLEAN REGARDING WEIGHTS FOR METRICS CALC :
    # commented out, put back if needed

    # weights = np.array([config["classes"][i][0] for i in config["classes"]])
    # unused_classes = np.where(weights == 0)[0]
    # confmat_cleaned = np.delete(sum_confmat, unused_classes, axis=0)  # remove rows
    # confmat_cleaned = np.delete( confmat_cleaned, unused_classes, axis=1

    per_c_ious, avg_ious = class_IoU(sum_confmat)
    ovr_acc = overall_accuracy(sum_confmat)
    per_c_precision, avg_precison = class_precision(sum_confmat)
    per_c_recall, avg_recall = class_recall(sum_confmat)
    per_c_fscore, avg_fscore = class_fscore(sum_confmat)

    # dict with the metrics
    # key : input name
    # value : dict with the metrics

    metrics = {}

    # write in out_csv like :
    # key : {model name
    # model size
    # patch size
    # stride
    # padding
    # stitching method
    # margin
    # metric, value}

    pass


def compute_metrics_patch(config: dict) -> None:
    # get ground truth from config

    pass
