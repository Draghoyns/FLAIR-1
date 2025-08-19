"""inputs:
- data_paths.csv
- model checkpoints (model will be exported to onnx, saved, then used. TODO: If any error arises, switch to pytorch in a non-blocking way)

outputs:
- metrics.json
- TODO: predictions.csv (optional, if not saved, predictions will be kept in RAM)
"""

"""
Functions like :
- for each image in the dataset:
 - run the inference and TODO: keep on RAM (option for saving the predictions on disk)
 - compute the metrics and append to DF
- at the end, compute all metrics (average, std deviation, total) and save in a json file

"""

#### IMPORTS
import argparse
import datetime
import json
import os
import sys
import warnings

from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from codecarbon import OfflineEmissionsTracker

from src.zone_detect.inference import inference
from src.zone_detect.prepare import prepare_data, prepare_output
from src.zone_detect.model import prepare_model
from src.zone_detect.metrics.create_dataset import interactive_dataset
from src.zone_detect.metrics.metrics import add_confusion, process_metrics
from src.zone_detect.stitching_job import stitching
from src.zone_detect.utils import (
    extract_method,
    gen_param_combination,
    setup,
    setup_out_path,
    timer,
    Logger,
)

Config = dict[str, Any]

warnings.simplefilter(action="ignore", category=FutureWarning)

#### PARSER
argParser = argparse.ArgumentParser()
argParser.add_argument(
    "--data",
    help="path to the data paths csv file, structured as : input_img_path, truth_path",
)
argParser.add_argument("--ckpt", help="path to checkpoint file for swin-upernet")


#### UTILS
def set_config(args, arguments: dict[str, str]) -> Config:
    # TODO
    """Given the inputs, set the elements in the config dict."""
    # input_root: / + dpt + zone
    # input_img_path: .tif
    # input_path : directory if multiple inputs

    # truth_root: / + dpt + zone
    # truth_path : .tif

    # get data type
    data_type = get_datatype_from_ckpt(args.ckpt)
    conf = f"src/zone_detect/metrics/configs/frozen_config_{data_type}.yaml"
    arguments["conf"] = conf

    # load config file and set up device
    config, device, use_gpu = setup(arguments)

    data = args.data
    if not data:
        data = interactive_dataset()

    # set paths
    config["data_paths"] = data
    config["model_weights"] = args.ckpt

    # set model
    # config.update({"sparse": SPARSE})
    config = prepare_model(config, device)

    # set output paths
    someparam = "_" + str(config.get("sparse", 0.0))
    if someparam == "_0.0":
        someparam = ""

    model_nickname = config["model_name"].split("-")[-1]
    model_type = config.get("model_type", "type-unknown")
    device_type = "gpu" if use_gpu else "cpu"
    date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    new_folder = f"{date}_{model_nickname}_{model_type}_{device_type}{someparam}"

    config.update(
        {
            "output_path": f"{config['output_path']}/{new_folder}",
            "metrics_out": f"{config['output_path']}/{new_folder}/metrics.json",
        }
    )

    return config


def load_csv(data_path: str) -> pd.DataFrame:
    """Loads a CSV file into a pandas DataFrame and verifies required columns.

    Parameters:
        data_path (str): Path to the CSV file containing image paths.

    Returns:
        pd.DataFrame: DataFrame containing the CSV data with at least 'input_img_path' and 'truth_path' columns.

    Raises:
        ValueError: If the CSV does not contain 'input_img_path' or 'truth_path' columns.
    """

    df = pd.read_csv(data_path)
    if "input_img_path" not in df.columns or "truth_path" not in df.columns:
        raise ValueError("CSV must contain 'input_img_path' and 'truth_path' columns.")
    return df


def prepare_np_output(profile: dict, config: Config, identifier: str) -> np.ndarray:

    output_type = config["output_type"]
    n_classes = config["n_classes"]

    # if we ever need to save
    # config, path_out = setup_indiv_path(config, identifier)

    if output_type == "argmax":
        pred_shape = (2, profile["height"], profile["width"])
    else:
        pred_shape = (n_classes, profile["height"], profile["width"])

    image_predictions = np.zeros(pred_shape, dtype=np.uint8)

    return image_predictions


def single_run_pipeline(
    param_combi: dict[str, Any],
    config: Config,
    metrics_df: pd.DataFrame,
    confmat_dict: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Runs the inference pipeline for a single parameter combination on one image.
    Args:
        param_combi (dict): Dictionary containing the specific method parameters for the inference.
        config (Config): Configuration dictionary.
        metrics_df (pd.DataFrame): DataFrame to store metrics (global, indexed by method).
        confmat_dict (dict[str, np.ndarray]): Confusion matrix for metrics (specific to one method).
    Returns:
        tuple: Updated metrics DataFrame and confusion matrix.
    """
    timings = {}

    img_pixels_detection = param_combi["img_pixels_detection"]
    margin = param_combi["margin"]
    padding = param_combi["padding"]
    stride = param_combi["stride"]
    stitch = param_combi["stitching"]

    output_type = config["output_type"]
    n_classes = config["n_classes"]
    model_type = config.get("model_type", "pytorch")
    model_args = config.get("model_args", dict())

    effective_output_type = output_type if stitch == "exact-clipping" else "class_prob"

    config.update(
        {
            "img_pixels_detection": img_pixels_detection,
            "margin": margin,
            "padding": padding,
            "stride": stride,
            "stitching": stitch,
            "effective_output_type": effective_output_type,
        }
    )

    method = f"size={img_pixels_detection}_stride={stride}_margin={margin}_padding={padding}_stitching={stitch}"
    identifier = "_" + method

    # start timer
    method_start = datetime.datetime.now()

    timed_prepare_data = timer(timings)(prepare_data)

    dataset, data_loader, sliced_dataframe, profile = timed_prepare_data(config)

    data_prep_time = timings["prepare_data"]

    single_area = sliced_dataframe["geometry"].area.sum()

    # prepare output raster
    np_predictions = prepare_np_output(profile, config, identifier)

    out, path_out = prepare_output(
        config,
        profile,
        identifier,
    )

    pure_infer_time = 0  # ms
    data_write_time = 0  # ms

    timed_inference = timer(timings)(inference)

    #### INFERENCE
    print(f"""    [ ] starting inference...\n""")
    for samples in tqdm(data_loader, ncols=75):

        predictions, indices = timed_inference(
            model_type=model_type,
            config=config,
            args=model_args,
            samples=samples,
        )

        pure_infer_time += timings["inference"]

        # writing windowed raster to output raster
        timer_write = datetime.datetime.now()

        for prediction, index in zip(predictions, indices):

            # stitching method is handled inside
            prediction, window = stitching(
                param_combi,
                sliced_dataframe,
                prediction,
                index,
                out,
                config["effective_output_type"],
            )

            prediction_to_write = prediction.copy()
            prediction_to_write[1:] = prediction[1:] * 65535
            prediction_to_write = prediction_to_write.astype("uint16")

            # write
            if output_type == "argmax":
                out.write_band([1, 2], prediction_to_write, window=window)
            else:
                out.write_band(
                    [i for i in range(1, n_classes + 1)],
                    prediction_to_write,
                    window=window,
                )
            data_write_time += (
                datetime.datetime.now() - timer_write
            ).total_seconds() * 1000  # ms

    out.close()
    dataset.close_raster()  # type: ignore

    #### METRICS
    # add confusion matrix for metrics
    metrics_matrix = confmat_dict.get(method, np.zeros((n_classes, n_classes)))
    metrics_matrix = add_confusion(
        Path(path_out),
        config["truth_path"],
        metrics_matrix,
        n_classes,
        stride,
    )
    confmat_dict.update({method: metrics_matrix})

    # end of processing
    ### timing
    total_time = (datetime.datetime.now() - method_start).total_seconds() * 1000  # ms

    # time metrics structured as follows:
    single_times_area = {
        "data_prep_time": data_prep_time,
        "pure_infer_time": pure_infer_time,
        "data_write_time": data_write_time,
        "total_time": total_time,
        "patches": len(sliced_dataframe),
        "area": single_area,
        "method": method,
    }

    # append metrics to dataframe
    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([single_times_area])], ignore_index=True
    )

    # delete inference image and log file (should be able to just not save them at all)
    os.remove(path_out)

    return metrics_df, confmat_dict


def run_pipeline(
    config: Config, metrics_df: pd.DataFrame, confmat_dict: dict[str, np.ndarray]
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Works for a single input image"""

    # set up common output path
    config = setup_out_path(config)

    # extracting config parameters
    local_out = Path(config["local_out"])
    save_logs = config.get("save_logs", False)

    #### LOGGING
    # TODO: do not save log and make in-terminal log less bloated
    log_filename = local_out / Path(
        f"{config['output_name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    if save_logs:
        sys.stdout = Logger(filename=str(log_filename))
        sys.stderr = sys.stdout
        print(f"    [LOGGER] Writing logs to: {log_filename}")
    else:
        print(
            f"    [INFO] Running inference for {config['output_name']} (logs in terminal only)"
        )

    #### SETUP
    settings = gen_param_combination(config, True)

    for combi in settings:

        metrics_df, confmat_dict = single_run_pipeline(
            param_combi=combi,
            config=config,
            metrics_df=metrics_df,
            confmat_dict=confmat_dict,
        )

    sys.stdout = original_stdout
    sys.stderr = original_stderr

    return metrics_df, confmat_dict


def batch_pipeline(config: Config) -> None:
    """
    Compute metrics for a batch of images.
    Args:
        gt_dpt (Path): Path to the ground truth directory of the department.
        config (dict): Configuration, in which the parameters for the inference are specified
    """

    # carbon tracking
    tracker = OfflineEmissionsTracker(
        country_iso_code="FRA", measure_power_secs=1e9, log_level="critical"
    )
    tracker.start()

    path_df = load_csv(config["data_paths"])

    # initialize metrics related variables for aggregation
    confmat_dict = {}

    csv_path = os.path.join(
        config["output_path"], "metrics.csv"
    )  # just in case df doesn't fit in RAM

    metrics_df = pd.DataFrame(
        columns=[
            "data_prep_time",
            "pure_infer_time",
            "data_write_time",
            "total_time",
            "patches",
            "area",
            "method",
        ]
    )

    for row in path_df.itertuples(index=False):
        img_path = str(row.input_img_path)
        truth_path = str(row.truth_path)

        config.update(
            {
                "truth_path": truth_path,
                "input_img_path": img_path,
                "output_name": f"{Path(img_path).stem}-ARGMAX",
            }
        )

        # __________INFERENCE__________#
        metrics_df, confmat_dict = run_pipeline(
            config, metrics_df=metrics_df, confmat_dict=confmat_dict
        )

    out_json = config.get("metrics_out", "metrics.json")
    out = Path(out_json).with_suffix(".json")

    emissions = tracker.stop()
    if emissions is None:
        emissions = 0.0

    metrics_file = []

    for method, metrics_matrix in confmat_dict.items():

        info = extract_method(method)
        method_df = metrics_df.query(f"method == '{method}'")
        nb_patches = method_df["patches"].to_list()
        times = method_df[
            ["data_prep_time", "pure_infer_time", "data_write_time", "total_time"]
        ].to_dict(orient="list")

        metrics_file.append(
            process_metrics(
                confmat=metrics_matrix,
                config=config,
                info=info,
                method_times=times,
                method_patches=nb_patches,
                area=method_df["area"].sum(),
                carbon=emissions,
            )
        )

    # save the metrics to a json file
    with open(out, "w") as f:
        json.dump(metrics_file, f, indent=2)

    print(f"Metrics saved to {out}")


def get_datatype_from_ckpt(ckpt: str) -> str:
    """Extracts the data type from the checkpoint name."""
    print(
        "\n[!] Make sure the checkpoint name contains the data type, that is 'IRC' or 'RVB'."
    )
    ckpt = ckpt.lower()
    if "irc" in ckpt or "ir-r-g" in ckpt and "rvb" not in ckpt and "rvb" not in ckpt:
        return "IRC"
    elif "rvb" in ckpt or "rgb" in ckpt:
        return "RVB"
    else:
        print(
            "WARNING: No data type found in the checkpoint name, defaulting to IRC. "
            "Please ensure the checkpoint name contains 'IRC' or 'RVB'."
        )
        return "IRC"


def main():

    # get data paths and model ckpt
    args = argParser.parse_args()
    arguments = {"conf": ""}

    # Set up the config
    config = set_config(args, arguments)

    batch_pipeline(config)


if __name__ == "__main__":

    main()

# command to run the script:
# update this depending on the paths wanted

#### SWIN IRC
# python src/zone_detect/metrics/main.py --data=0testing_saves/data_paths_IRC.csv --ckpt=/media/DATA/INFERENCE_HS/MODELS_IA/FLAIR1/swin-upernet-small_IRV_SET1/checkpoints/ckpt-epoch=84-val_loss=0.37_00_HF_SwinUpernet_Small_IR-R-G_set1.ckpt

#### SWIN RVB
# python src/zone_detect/metrics/main.py --data=0testing_saves/data_paths_RVB.csv --ckpt=/var/tmp/shys/INFERENCE_HS/MODELS_IA/FLAIR1/swin_upernet_small_RVB/ckpt-epoch=126-val_loss=0.37_03-swin-upernet-small_PROD_FLAIR-INC_RVB_ini-none_custom.ckpt


#### UNET
# python src/zone_detect/metrics/main.py --data=0testing_saves/data_paths_RVB.csv --ckpt=/media/DATA/INFERENCE_HS/MODELS_IA/FLAIR1/unet_resnet/FLAIR-INC_rgb_15cl_resnet34-unet_weights.pth

# python src/zone_detect/metrics/main.py --data=0testing_saves/data_paths_IRC.csv --ckpt=/

# /var/tmp/shys/INFERENCE_HS/MODELS_IA/FLAIR1/swin-upernet-small_IRV_SET1/checkpoints/ckpt-epoch=84-val_loss=0.37_00_HF_SwinUpernet_Small_IR-R-G_set1.ckpt
