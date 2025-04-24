import datetime
import numpy as np
from pathlib import Path


def overall_accuracy(npcm):
    oa = np.trace(npcm) / npcm.sum()
    return 100 * oa


def class_IoU(npcm, n_class):
    ious = (
        100
        * np.diag(npcm)
        / (np.sum(npcm, axis=1) + np.sum(npcm, axis=0) - np.diag(npcm))
    )
    ious[np.isnan(ious)] = 0
    return ious, np.mean(ious)


def class_fscore(precision, recall):
    fscore = 2 * (precision * recall) / (precision + recall)
    fscore[np.isnan(fscore)] = 0
    return fscore, np.mean(fscore)


def compute_metrics(config: dict) -> None:

    # get output path from config
    if config["metrics_out"]:
        out_csv = config["metrics_out"]
    else:
        # date and time identifier
        out_csv = (
            config["output_path"]
            + "/metrics_"
            + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # type: ignore
            + ".csv"
        )
    # get folder with saves from config
    preds_folder = (
        config["output_path"] + "/"
        if config["output_path"][-1] != "/"
        else config["output_path"]
    )
    preds_folder += config["output_name"]
    preds_folder = Path(preds_folder)

    # get ground truth from config
    truth_msk = config["truth_path"]

    for file in preds_folder.iterdir():
        if file.is_file() and file.suffix == ".tif":
            pass
    # for each save in the folder
    # compare to truth
    # dict with the metrics
    # key : date + time
    # value : dict with the metrics

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
