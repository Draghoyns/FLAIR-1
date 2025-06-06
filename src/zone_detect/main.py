import argparse
import datetime
import json
import sys
from tqdm import tqdm
import warnings

from pathlib import Path
from typing import Any

from codecarbon import OfflineEmissionsTracker

from geopandas import GeoDataFrame

from pytorch_lightning.utilities.rank_zero import rank_zero_only  # type: ignore

import rasterio

import onnxruntime as ort

import torch
from torch.utils.data import DataLoader

from src.zone_detect.dataset import Sliced_Dataset
from src.zone_detect.inference import inference
from src.zone_detect.model import load_model
from src.zone_detect.slicing_job import slice_extent, slice_extent_separate
from src.zone_detect.stitching_job import stitching

from src.zone_detect.test.metrics import batch_metrics, compute_metrics_patch
from src.zone_detect.test.onnx.onnx_export import get_onnx_path
from src.zone_detect.test.tiles import get_stride

from src.zone_detect.utils import (
    gen_param_combination,
    open_images,
    setup_device,
    setup_out_path,
    setup,
    setup_indiv_path,
)

Config = dict[str, Any]

warnings.simplefilter(action="ignore", category=FutureWarning)

#### PARSER
argParser = argparse.ArgumentParser()
argParser.add_argument("--conf", help="path to the .yaml config file")
argParser.add_argument(
    "-c",
    "--compare",
    action="store_true",
    help="compare different methods",
)
argParser.add_argument("-m", "--metrics", action="store_true", help="compute metrics")
argParser.add_argument(
    "-b", "--batch_mode", action="store_true", help="run on a batch of input images"
)
argParser.add_argument(
    "-o", "--onnx", action="store_true", help="use ONNX model instead of PyTorch"
)


# __________Logging___________#
@rank_zero_only
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        self.encoding = self.terminal.encoding

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


# we're not handling multiple inputs yet
def conf_log(
    config: Config,
    resolution: tuple[float, float],
    img_size: tuple[int, int],
) -> None:
    # Determine model template info based on provider
    mf = config["model_framework"]
    provider = mf["model_provider"]
    if provider == "HuggingFace":
        model_template = f"{provider} - {mf['HuggingFace']['org_model']}"
    elif provider == "SegmentationModelsPytorch":
        model_template = (
            f"{provider} - {mf['SegmentationModelsPytorch']['encoder_decoder']}"
        )
    else:
        model_template = provider  # fallback if unknown

    compare_handling = "strategies" in config
    compare = config["compare"]
    strategies = config["strategies"]

    if compare:
        compare_param = f"""
    |- overlapping strategy: {"handled" if compare_handling  else "exact"}
    |- tiling comparison: {"yes" if (compare_handling and strategies['tiling']['enabled']) else "no"}
    |- stitching comparison: {"no" if not compare_handling else strategies['stitching']['method']}
    |- padding: {"not handled" if not compare_handling else strategies['padding_overall']} \n """
    else:
        compare_param = ""

    print("    [ ] no comparison" if not compare else "    [x] comparison")
    log = [
        f"""
    |- output path: {config['output_path']}
    |- output raster name: {config['output_name']}

    |- input image path: {config['input_img_path']}
    |- channels: {config['channels']}
    |- input image WxH: {img_size}   
    |- resolution: {resolution}
    |- write dataframe: {config['write_dataframe']}
    |- number of classes: {config['n_classes']}
    |- normalization: {config['norma_task'][0]['norm_type']}
    |- output type: {config['output_type']}

    |- model weights path: {config['model_weights']}
    |- model template: {model_template}
    |- device: {"cuda" if config['use_gpu'] else "cpu"}
    |- batch size: {config['batch_size']}
    """
    ]
    print("\n".join(log + [compare_param]))


# __________Prepare objects___________#
def prepare_tiles(
    config: Config,
    stride: int,
) -> tuple[GeoDataFrame, dict, tuple[float, float]]:
    """Slicing extent for overlapping detection"""
    input_path = Path(config["input_img_path"])
    patch_size = config["img_pixels_detection"]
    margin = config["margin"]
    output_name = config["output_name"]
    output_path = Path(config["local_out"])
    write_df = config["write_dataframe"]

    sliced_dataframe, profile, resolution, img_size = slice_extent(
        in_img=input_path,
        patch_size=patch_size,
        margin=margin,
        output_name=output_name,
        output_path=output_path,
        write_dataframe=write_df,
        stride=stride,
    )
    ## log
    conf_log(config, resolution, img_size)
    print(f"""    [x] sliced input raster to {len(sliced_dataframe)} squares...""")

    return sliced_dataframe, profile, resolution


def prepare_data(
    config: Config, stride: int
) -> tuple[Sliced_Dataset, DataLoader, GeoDataFrame, dict]:

    # slicing
    sliced_dataframe, profile, resolution = prepare_tiles(config, stride)

    # get dataset
    dataset = Sliced_Dataset(
        dataframe=sliced_dataframe,
        img_path=config["input_img_path"],
        resolution=resolution,
        bands=config["channels"],
        patch_detection_size=config["img_pixels_detection"],
        norma_dict=config["norma_task"],
    )

    # get Dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_worker"],
        pin_memory=True,
    )

    return dataset, data_loader, sliced_dataframe, profile


def prepare_model(config: Config, device: torch.device) -> tuple[str, dict[str, Any]]:
    print(
        f"""
    ##############################################
    ZONE DETECTION
    ##############################################
    """
    )

    onnx = config["onnx"]
    arg_package = dict()

    if onnx:
        model_type = "onnx"
        print(f"""    [ ] using ONNX model...""")

        providers = {"gpu": "CUDAExecutionProvider", "cpu": "CPUExecutionProvider"}

        # get existing path or export
        path = get_onnx_path(config)
        ort_session = ort.InferenceSession(path, providers=[providers["gpu"]])

        arg_package.update(
            {
                "ort_session": ort_session,
            }
        )

    else:
        model_type = "pytorch"
        print(
            f"""    [ ] using PyTorch model...

        CUDA available? {torch.cuda.is_available()}
        """
        )

        ## loading model and weights
        model = load_model(config)
        model.eval()
        model = model.to(device)
        print(f"""    [x] loaded model and weights...""")

        arg_package.update(
            {"model": model, "device": device, "use_gpu": config["use_gpu"]}
        )

    return model_type, arg_package


def prepare_output(
    config: Config,
    profile: dict,
    identifier: str = "",
) -> tuple[rasterio.io.DatasetWriter, str]:  # type: ignore
    """Prepare output raster profile and output path"""

    config, path_out = setup_indiv_path(config, identifier)
    size = config["img_pixels_detection"]

    out_profile = profile.copy()
    out_profile.update(
        {
            "dtype": "uint8",
            "compress": "LZW",
            "driver": "GTiff",
            "BIGTIFF": "YES",
            "tiled": True,
            "blockxsize": size,
            "blockysize": size,
        }
    )
    out_profile["count"] = (
        2 if config["output_type"] == "argmax" else config["n_classes"]
    )

    # second band gives the max probability

    out = rasterio.open(path_out, "w+", **out_profile)
    return out, path_out


# _________PIPELINES__________#
def run_from_config(config: Config) -> None:
    """Run the pipeline from a config file"""
    # setting up device and log
    device, use_gpu = setup_device(config)

    run_pipeline(config, device, use_gpu)


def run_pipeline(config: Config, device: torch.device, use_gpu: bool) -> None:
    """Works for a single input image"""

    # set up common output path
    config = setup_out_path(config)

    # extracting config parameters
    output_type = config["output_type"]
    n_classes = config["n_classes"]
    compare = config["compare"]
    local_out = Path(config["local_out"])
    compute_metrics = config["metrics"]

    # log
    log_filename = local_out / Path(
        f"{config['output_name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    sys.stdout = Logger(filename=str(log_filename))
    sys.stderr = sys.stdout
    print(f"    [LOGGER] Writing logs to: {log_filename}")

    # model
    model_type, model_args = prepare_model(config, device)

    # setup elements for the metrics
    truth_array, metrics_json = open_images(
        config,
        local_out,
        compute_metrics,
    )

    if compare:

        method_times = {}

        print(f"""    [ ] starting comparison...\n""")

        settings = gen_param_combination(config)
        for combi in settings:

            method_metrics = []

            img_pixels_detection = combi["img_pixels_detection"]
            margin = combi["margin"]
            padding = combi["padding"]
            stride = combi["stride"]
            stitch = combi["stitching"]

            config.update(
                {
                    "img_pixels_detection": img_pixels_detection,
                    "margin": margin,
                    "padding": padding,
                    "stride": stride,
                    "stitching": stitch,
                }
            )

            method = f"size={img_pixels_detection}_stride={stride}_margin={margin}_padding={padding}_stitching={stitch}"
            identifier = "_" + method

            # start timer
            start_time = datetime.datetime.now()

            dataset, data_loader, sliced_dataframe, profile = prepare_data(
                config, stride
            )
            # prepare output raster
            out, path_out = prepare_output(
                config,
                profile,
                identifier,
            )
            print(f"""    [ ] starting inference...\n""")
            for samples in tqdm(data_loader):

                predictions, indices = inference(
                    model_type=model_type,
                    config=config,
                    args=model_args,
                    samples=samples,
                )
                # writing windowed raster to output raster
                for prediction, index in zip(predictions, indices):

                    # stitching method is handled inside
                    prediction, window = stitching(
                        config,
                        sliced_dataframe,
                        prediction,
                        index,
                        out,
                        stitch,
                        stride,
                    )
                    # write
                    if output_type == "argmax":
                        out.write_band([1, 2], prediction, window=window)
                    else:
                        out.write_band(
                            [i for i in range(1, n_classes + 1)],
                            prediction,
                            window=window,
                        )

                    if compute_metrics:
                        # compute metrics per patch
                        inference_time = (
                            datetime.datetime.now() - start_time
                        ).total_seconds() * 1000  # ms                        inference_time = inference_time.total_seconds()
                        if method not in method_times:
                            method_times[method] = [inference_time]
                        else:
                            method_times[method].append(inference_time)

                        method_metrics.append(
                            compute_metrics_patch(
                                prediction,
                                truth_array,
                                window,
                                config,
                                method,
                            )
                        )

            out.close()
            dataset.close_raster()  # type: ignore

            print(
                f"""    [X] done writing to {path_out.split('/')[-1]} raster file.\n"""
            )

            if compute_metrics:
                config["times"] = method_times
                print(
                    f"""    [X] done writing metrics to {metrics_json.name} file.\n"""
                )

                with open(metrics_json, "w") as f:
                    json.dump(method_metrics, f, indent=2)

    else:

        # default configuration : exact clipping and default sized tiling

        stride = get_stride(config)[0]
        dataset, data_loader, sliced_dataframe, profile = prepare_data(config, stride)

        # prepare output raster
        out, path_out = prepare_output(config, profile)
        # inference loop

        print(f"""    [ ] starting inference...\n""")
        for samples in tqdm(data_loader):

            predictions, indices = inference(
                model_type=model_type,
                args=model_args,
                config=config,
                samples=samples,
            )

            # writing windowed raster to output rastert
            for prediction, index in zip(predictions, indices):

                prediction, window = stitching(
                    config,
                    sliced_dataframe,
                    prediction,
                    index,
                    out,
                    "exact-clipping",
                    stride,
                )
                # write
                if output_type == "argmax":
                    out.write_band([1, 2], prediction, window=window)
                else:
                    out.write_band(
                        [i for i in range(1, n_classes + 1)], prediction, window=window
                    )

        out.close()
        print(
            f"""    
                        
            [X] done writing to {path_out.split('/')[-1]} raster file.\n"""
        )

    dataset.close_raster()  # type: ignore

    sys.stdout = sys.__stdout__


def batch_metrics_pipeline(
    config: Config, truth_dpt: Path, device: torch.device, use_gpu: bool
) -> None:
    """
    Compute metrics for a batch of images.
    Args:
        gt_dpt (Path): Path to the ground truth directory of the department.
        config (dict): Configuration, in which the parameters for the inference are specified
    """

    out_json = config["metrics_out"]
    data_type = config["data_type"]
    file_pattern = f"*{data_type}.tif"
    compute_metrics = config["metrics"]

    # output file
    if compute_metrics:
        assert out_json, "Please provide an output path for the metrics"

    # __________INFERENCE__________#
    inputs_dpt = Path(config["input_path"])

    zone_list = sorted(p for p in inputs_dpt.iterdir() if p.is_dir())
    for full_zone in zone_list:

        # find an input file image
        img_path = next(full_zone.glob(file_pattern), None)
        if img_path is None:
            continue

        if compute_metrics:
            dpt, zone = img_path.parts[-3:-1]
            truth_dir = truth_dpt / zone
            truth_path = next(Path(truth_dir).glob("*.tif"), None)
            if truth_path is None:
                print(f"No ground truth found for zone: {zone}")
                continue
            config.update({"truth_path": str(truth_path)})

        config.update(
            {
                "input_img_path": str(img_path),
                "output_name": f"D{img_path.stem}-ARGMAX-S",
            }
        )

        # Inference and saving the predictions
        run_pipeline(config, device, use_gpu)

    # we have all the predictions in the output folder

    if compute_metrics:

        out = Path(out_json).with_suffix(".json")

        metrics_file = batch_metrics(config, truth_dpt)

        # save the metrics to a json file
        json.dump(
            metrics_file,
            open(out, "w"),
        )
        print(f"Metrics saved to {out}")


# __________Main function___________#
def main():

    tracker = OfflineEmissionsTracker(country_iso_code="FRA", measure_power_secs=1e9)
    tracker.start()

    # reading yaml
    args = argParser.parse_args()

    # setting up device and log
    config, device, use_gpu = setup(args)

    if args.batch_mode:
        gt_dir = Path(config["truth_root"])
        gt_dpt = gt_dir / Path(config["truth_path"]).parts[-3]

        batch_metrics_pipeline(config, gt_dpt, device, use_gpu)
    else:
        run_pipeline(config, device, use_gpu)

    emissions = tracker.stop()

    print(f"Emissions: {emissions} g CO2")


if __name__ == "__main__":

    main()
