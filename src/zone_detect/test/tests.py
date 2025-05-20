# testing all sorts of functions for my own sanity

import json
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from src.zone_detect.test.metrics import *
from src.zone_detect.utils import extract_method, info_extract


def test_error_rate(img_path: Path, out_dir: Path, verbose: bool = False) -> None:

    # for two identical images, the error rate should be 0 everywhere

    print(
        """Test for error_rate_patch : 
    - Testing identical images"""
    )

    error = error_rate_patch(img_path, out_dir, img_path, dic={}, save=True)
    error = error["img_path"]

    if not np.any(error):
        print("     [x] OK")
    else:
        if verbose:
            print(
                f"""     [ ] ERROR
                The output looks like this : {error[:6, :6]}"""
            )
        print(
            f"""     [ ] ERROR
        Go check the output at : {out_dir}"""
        )


def test_error_rate_patch(
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

    print(f"Target shape: {target.shape}")
    print(f"Pred shape: {pred.shape}")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out_path = out_path / f"error_rate_{full_method}_{datetime.datetime.now()}.png"

    # better visualization
    from scipy.ndimage import gaussian_filter

    out_array = gaussian_filter(out_array, sigma=2)

    # save the error rate as a png
    autoscale = True
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


def test_error_rate_loop(truth_dir: Path, out_dir: Path, pred_dir: Path) -> None:
    """Args:
    pred_dir (Path): the directory with predictions
    out_dir (Path): the output directory for the error rate
    truth_dir (Path): the ground truth directory of the department"""
    dic = {}

    # get all tif files in the pred_dir
    tif_files = list(pred_dir.rglob("*.tif"))

    # Check result for identical images
    # img = tif_files[0]
    # test_error_rate(img, out_dir)
    # ________________________________#

    for pred_path in tqdm(tif_files, desc="Computing error rate"):

        # get corresponding truth file
        truth_file = get_truth_path(pred_path, truth_dir)

        dic = test_error_rate_patch(
            truth_file=truth_file,
            out_dir=out_dir,
            pred_path=pred_path,
            dic=dic,
            save=True,
        )

    # aggregate the error rate for each method over all kays
    methods = dict()
    total = dict()
    for key in dic.keys():
        # get the method name
        method = info_extract(Path(key))["method"]
        size = info_extract(Path(key))["patch_size"]
        if size != 1024:
            continue

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


# last checked and updated : 20250519 1218
def test_batch_metrics_pipeline(config: dict, truth_dpt: Path) -> None:
    """
    Compute metrics for a batch of images.
    Args:
        gt_dpt (Path): Path to the ground truth directory of the department.
        config (dict): Configuration, in which the parameters for the inference are specified
    """

    print(
        """Testing batch metrics pipeline...
    ______________________________
    [ ] Sanity check of accessing files
    """
    )

    out_json = Path(config["output_path"] + "/metrics.json")
    data_type = config.get("data_type")
    file_pattern = f"*{data_type}.tif"
    # out_json = Path(config["metrics_out"])

    # output file
    assert out_json, "Please provide an output path for the metrics"

    # __________INFERENCE__________#
    inputs_dpt = Path(config["input_path"])
    print(f"Input path: {inputs_dpt}")

    zone_list = sorted(p for p in inputs_dpt.iterdir() if p.is_dir())
    print(f"Zone list: {zone_list}")

    for full_zone in zone_list:

        # find an input file image
        irc_path = next(full_zone.glob(file_pattern), None)
        if irc_path is None:
            continue

        print(f"Processing image: {irc_path}")

        dpt, zone = irc_path.parts[-3:-1]
        truth_dir = truth_dpt / zone
        truth_path = next(Path(truth_dir).glob("*.tif"), None)
        if truth_path is None:
            print(f"No ground truth found for zone: {zone}")
            continue
        print(f"Ground truth path: {truth_path}")

        config.update(
            {
                "input_img_path": str(irc_path),
                "truth_path": str(truth_path),
                "output_name": f"{irc_path.stem}-ARGMAX-S",
            }
        )

    print(
        """     [x] OK
    ______________________________
    Here the inference is run.
    
    [ ] Checking metrics computation
    """
    )

    # we have all the predictions in the output folder

    out = out_json.with_suffix(".json")
    print(f"Output path for metrics: {out}")

    metrics_file = batch_metrics(config, truth_dpt)

    # save the metrics to a json file
    json.dump(
        metrics_file,
        open(out, "w"),
    )
    print(f"Metrics saved to {out}")


# last checked and updated : 20250519 1205
def test_batch_metrics(config: dict, truth_dir: Path) -> list:
    print(
        """Testing batch metrics...
    ______________________________"""
    )
    metrics_file = []

    df = collect_paths_truth(config, truth_dir)
    classes = config["classes"]
    n_classes = len(classes)
    print(f"Classes are: {classes}")

    print(f"Dataframe looks like: {df.head(5)}")
    print(f"Dataframe methods are: {df['method'].unique()}")

    grouped = df.groupby("method")

    # metrics for each method
    print("Computing metrics...")
    for method, group in grouped:

        pred_paths = group["pred_path"].tolist()
        gt_paths = group["truth_path"].tolist()

        patch_confusion_matrices = []
        sum_confmat = np.zeros((n_classes, n_classes))
        print(f"Processing method: {method}")

        for pred_path, truth_path in tqdm(
            zip(pred_paths, gt_paths), desc="Processing images"
        ):
            try:
                # loading
                with rasterio.open(pred_path) as src:
                    preds = src.read(1)
                with rasterio.open(truth_path) as src:
                    target = src.read(1) - 1

                # patch_confusion_matrices.append( confusion_matrix( target.flatten(), preds.flatten(), labels=list(range(int(len(config["classes"])))) + [255],))
                sum_confmat += faster_confusion_matrix(
                    target.flatten(), preds.flatten(), n_classes
                )

            except Exception as e:
                print(f"Error processing {pred_path} and {truth_path}: {e}")

        # compute metrics for the group
        # sum_confmat = np.sum(patch_confusion_matrices, axis=0)
        confmat_cleaned = clean_confmat(sum_confmat, config)

        # metrics
        with np.errstate(divide="ignore", invalid="ignore"):
            # nans are handled dont worry
            per_c_ious, avg_ious = class_IoU(confmat_cleaned)
            ovr_acc = overall_accuracy(confmat_cleaned)
            per_c_fscore, avg_fscore = class_fscore(confmat_cleaned)
            method_times = config.get("times", {}).get(method, [])
            avg_time = np.mean(method_times) if method_times else 0

        print("Storing metrics...")

        # method parameters
        # mock_path = Path("/region/" + str(method) + ".tif")
        # info = info_extract(mock_path)
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
            "Avg_metrics_name": ["mIoU", "Overall Accuracy", "Fscore", "Time"],
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

    print(
        f"""Stats: 
    ______________________________
    Length : {len(metrics_file)}
    Sample : {metrics_file[:2]}
    """
    )

    return metrics_file


# last checked and updated : 20250519 1144
def test_collect_paths_truth(config: dict, truth_dir: Path) -> pd.DataFrame:
    path_collection = []

    # get predictions
    pred_dir = Path(config["output_path"])
    print(f"Looking for predictions in:  \n {pred_dir}")
    timed_folders = [p for p in pred_dir.iterdir() if p.is_dir()]
    print(f"Found {len(timed_folders)} folders")

    # dataframe with pred path, gt path and method single string
    for timestamp in sorted(timed_folders):
        pred_files = list(timestamp.rglob("*.tif"))
        print(f"Found {len(pred_files)} predictions in {timestamp}")
        if len(pred_files) == 0:
            print(f"No predictions found in {timestamp}")
            continue

        truth_path = get_truth_path(pred_files[0], truth_dir)
        print(f"Found ground truth: {truth_path}")
        if truth_path is None:
            print(f"No ground truth found for {pred_files[0]}")
            continue

        for pred_path in pred_files:
            method_id = info_extract(pred_path)["method"]
            # method_id = Path(pred_path.name.split("IRC-ARGMAX-S_")[1]).stem
            # dpt_zone-name_data-type-ARGMAX-S_size=128_stride=96_margin=32_padding=some-padding_stitching=exact-clipping
            path_collection.append(
                {
                    "pred_path": str(pred_path),
                    "truth_path": str(truth_path),
                    "method": method_id,
                }
            )
    return pd.DataFrame(path_collection)


def test_compute_metrics_patch(
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
    print(f"Target shape: {target.shape}")
    print(f"Target looks like: \n {target[:6, :6]}")
    print(f"Pred shape: {pred_patch.shape}")
    print(f"Pred[0] looks like: \n {pred_patch[0][:6, :6]}")

    assert (
        pred_patch.shape == target.shape
    ), "Shapes of ground truth and prediction must match"

    classes = config["classes"]
    n_classes = len(classes)

    #### compute metrics
    # confusion matrix
    confmat = faster_confusion_matrix(target.flatten(), pred_patch.flatten(), n_classes)

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
