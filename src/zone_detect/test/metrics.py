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
from src.zone_detect.utils import extract_method, info_extract


#### UTILS ####
def clean_confmat(confmat: np.ndarray, config: dict) -> np.ndarray:
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


def valid_truth(config: dict) -> Path:
    """Check if the ground truth path is valid and coherent with the input path :
    the zone should be the same in both paths.
    """
    truth_path = Path(config["truth_path"])
    # verify coherence with input path
    sanity_check = str(config["input_img_path"]).split("/")[-3:-1]  # zone
    truth_check = list(truth_path.parts[-3:-1])
    if truth_check != sanity_check:
        raise ValueError(
            f"Ground truth path {truth_path} does not match input path {config['input_img_path']}"
        )
    return Path(truth_path)


def get_truth_path(pred_path: Path, truth_dir: Path) -> Path:
    dpt, zone_name = info_extract(pred_path)["dpt"], info_extract(pred_path)["zone"]

    # corresponding ground truth
    # we consider gt_folder the overall folder
    truth_subdir = truth_dir / zone_name
    truth_path = next(truth_subdir.glob("*.tif"), None)
    if truth_path is None:
        raise FileNotFoundError(
            f"Ground truth file not found in {truth_subdir}. Please check the folder."
        )
    return truth_path


def collect_paths_truth(config: dict, truth_dir: Path) -> pd.DataFrame:
    path_collection = []

    # get predictions
    pred_dir = Path(config["output_path"])
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
    pred_patch: np.ndarray,
    truth: np.ndarray,
    window: Window,
    config: dict,
    method: str,
) -> dict:
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
    # confmat = faster_confusion_matrix(target.flatten(), pred_patch.flatten(), n_classes)
    confmat = confusion_matrix(
        target.flatten(), pred_patch.flatten(), labels=range(n_classes)
    )

    confmat_cleaned = clean_confmat(confmat, config)

    with np.errstate(divide="ignore", invalid="ignore"):
        # nans are handled don't worry
        per_c_ious, avg_ious = class_IoU(confmat_cleaned)
        ovr_acc = overall_accuracy(confmat_cleaned)
        per_c_fscore, avg_fscore = class_fscore(confmat_cleaned)

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
            "classes": [classes[i][1] for i in range(1, n_classes + 1)],
            "per_class_iou": list(per_c_ious),
            "per_class_fscore": list(per_c_fscore),
        }
    }
    return metrics


def batch_metrics(config: dict, truth_dir: Path) -> list:
    """Compute metrics for each method in the batch mode.
    The metrics are computed for the whole image, not per patch.
    Args:
        config (dict): Configuration, in which the parameters for the inference are specified
        truth_dir (Path): Path to the ground truth directory.
    Returns:
        metrics_file (list): List of dictionaries containing the metrics for each method.
    """

    metrics_file = []
    df = collect_paths_truth(config, truth_dir)
    classes = config["classes"]
    n_classes = len(classes)

    grouped = df.groupby("method")

    # metrics for each method
    print("Computing metrics...")
    for method, group in tqdm(grouped, desc="Computing metrics...", total=len(grouped)):

        pred_paths = group["pred_path"].tolist()
        gt_paths = group["truth_path"].tolist()

        sum_confmat = np.zeros((n_classes, n_classes))

        for pred_path, truth_path in zip(pred_paths, gt_paths):
            try:
                # loading
                with rasterio.open(pred_path) as src:
                    preds = src.read(1)
                with rasterio.open(truth_path) as src:
                    target = src.read(1) - 1

                sum_confmat += confusion_matrix(
                    target.flatten(), preds.flatten(), labels=range(n_classes)
                )
            except Exception as e:
                print(f"Error processing {pred_path} and {truth_path}: {e}")

        # compute metrics for the group
        confmat_cleaned = clean_confmat(sum_confmat, config)

        # metrics
        with np.errstate(divide="ignore", invalid="ignore"):
            # nans are handled dont worry
            per_c_ious, avg_ious = class_IoU(confmat_cleaned)
            ovr_acc = overall_accuracy(confmat_cleaned)
            per_c_fscore, avg_fscore = class_fscore(confmat_cleaned)

            method_times = config.get("times", {}).get(method, [])
            avg_time = np.mean(method_times) if method_times else 0

        # method parameters
        info = extract_method(str(method))

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
            "Avg_metrics_name": ["mIoU", "Overall Accuracy", "Fscore", "Time in ms"],
            "Avg_metrics": [
                avg_ious,
                ovr_acc,
                avg_fscore,
                avg_time,
            ],
            "classes": [classes[i][1] for i in range(1, n_classes + 1)],
            "per_class_iou": list(per_c_ious),
            "per_class_fscore": list(per_c_fscore),
        }
        metrics_file.append(metrics)

    return metrics_file


def error_rate_loop(truth_dir: Path, out_dir: Path, pred_dir: Path) -> None:
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
        )

    # aggregate the error rate for each method over all kays
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
        autoscale = True
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


# not incorporated in the pipeline, but maybe as option ?
def error_rate_patch(
    truth_file: Path, out_dir: Path, pred_path: Path, dic: dict, save: bool
) -> dict[str, np.ndarray]:
    """Compute the error rate per patch for a given input.
    You need to provide full paths
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

    # save the error rate as a png
    autoscale = False
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
            out_array, cmap="plasma", interpolation="nearest", vmin=vmin, vmax=vmax
        )
        plt.colorbar()
        plt.title("Error Rate for method : \n" + full_method)
        plt.savefig(str(out_path))
        plt.close()
        print(f"Error rate saved to {out_path}")

    dic[pred_path] = out_array

    return dic


#### ANALYSIS ####
def load_metrics_json(json_path: Path) -> dict:
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


if __name__ == "__main__":
    # compute a posteriori metrics = error rate

    truth_dir = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/labels_raster/FLAIR_19/D037_2021/"
    out_dir = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/swin-upernet-small/D037_2021/out2025020/error_rate_margin=0_swin_RVB"
    pred_dir = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/swin-upernet-small/D037_2021/out20250520_swin_RVB_last"

    error_rate_loop(Path(truth_dir), Path(out_dir), Path(pred_dir))
