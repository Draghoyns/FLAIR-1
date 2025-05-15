import json
from pathlib import Path
import os, sys
import datetime
import warnings
import rasterio
import argparse
from tqdm import tqdm

from geopandas import GeoDataFrame
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.rank_zero import rank_zero_only  # type: ignore

from src.zone_detect.slicing_job import slice_extent, slice_extent_separate
from src.zone_detect.model import load_model
from src.zone_detect.dataset import Sliced_Dataset
from src.zone_detect.test.metrics import batch_metrics, compute_metrics_patch
from src.zone_detect.test.tiles import get_stride
from src.zone_detect.compare import inference, stitching

from src.zone_detect.utils import (
    gen_param_combination,
    setup_device,
    setup_out_path,
    setup,
    setup_indiv_path,
)


warnings.simplefilter(action="ignore", category=FutureWarning)

#### CONF FILE
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
    config: dict,
    resolution: tuple[float, float],
    img_size: list[int],
) -> None:
    # Determine model template info based on provider
    provider = config["model_framework"]["model_provider"]
    if provider == "HuggingFace":
        model_template = (
            f"{provider} - {config['model_framework']['HuggingFace']['org_model']}"
        )
    elif provider == "SegmentationModelsPytorch":
        model_template = f"{provider} - {config['model_framework']['SegmentationModelsPytorch']['encoder_decoder']}"
    else:
        model_template = provider  # fallback if unknown

    compare_handling = "strategies" in config

    compare_param = f"""
    |- overlapping strategy: {"handled" if compare_handling  else "exact"}
    |- tiling comparison: {"yes" if (compare_handling and config['strategies']['tiling']['enabled']) else "no"}
    |- stitching comparison: {"no" if not compare_handling else config['strategies']['stitching']['method']}
    |- padding: {"not handled" if not compare_handling else config["strategies"]['padding_overall']} \n """

    print("    [ ] no comparison" if not config["compare"] else "    [x] comparison")
    print(
        f"""
    |- output path: {config['output_path']}
    |- output raster name: {config['output_name']}

    |- input image path: {config['input_img_path']}
    |- channels: {config['channels']}
    |- input image WxH: {img_size[0], img_size[1]}   
    |- resolution: {resolution}
    |- write dataframe: {config['write_dataframe']}
    |- number of classes: {config['n_classes']}
    |- normalization: {config['norma_task'][0]['norm_type']}
    |- output type: {config['output_type']}

    |- model weights path: {config['model_weights']}
    |- model template: {model_template}
    |- device: {"cuda" if config['use_gpu'] else "cpu"}
    |- batch size: {config['batch_size']}
    {compare_param if config["compare"] else ""}
    """
    )


# __________Prepare objects___________#
def prepare_tiles(
    config: dict,
    stride: int,
) -> tuple[GeoDataFrame, dict, tuple]:
    ## slicing extent for overlapping detection
    sliced_dataframe, profile, resolution, img_size = slice_extent(
        in_img=config["input_img_path"],
        patch_size=config["img_pixels_detection"],
        margin=config["margin"],
        output_name=config["output_name"],
        output_path=config["local_out"],
        write_dataframe=config["write_dataframe"],
        stride=stride,
    )
    ## log
    conf_log(config, resolution, img_size)
    print(f"""    [x] sliced input raster to {len(sliced_dataframe)} squares...""")

    return sliced_dataframe, profile, resolution


def prepare_data(
    config: dict, stride: int
) -> tuple[Sliced_Dataset, DataLoader, GeoDataFrame, dict, tuple]:

    channels = config["channels"]
    norma_task = config["norma_task"]
    batch_size = config["batch_size"]
    num_worker = config["num_worker"]
    input_img_path = config["input_img_path"]
    img_pixels_detection = config["img_pixels_detection"]

    # slicing
    sliced_dataframe, profile, resolution = prepare_tiles(config, stride)

    # get dataset
    dataset = Sliced_Dataset(
        dataframe=sliced_dataframe,
        img_path=input_img_path,
        resolution=resolution,
        bands=channels,
        patch_detection_size=img_pixels_detection,
        norma_dict=norma_task,
    )

    # get Dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_worker,
        pin_memory=True,
    )

    return dataset, data_loader, sliced_dataframe, profile, resolution


def prepare_model(config: dict, device: torch.device) -> torch.nn.Module:

    print(
        f"""
    ##############################################
    ZONE DETECTION
    ##############################################

    CUDA available? {torch.cuda.is_available()}"""
    )

    ## loading model and weights
    model = load_model(config)
    model.eval()
    model = model.to(device)
    print(f"""    [x] loaded model and weights...""")

    return model


def prepare_output(
    config: dict,
    profile: dict,
    identifier: str = "",
) -> tuple[rasterio.io.DatasetWriter, str]:  # type: ignore
    """Prepare output raster profile and output path"""

    config, path_out = setup_indiv_path(config, identifier)
    output_type = config["output_type"]
    n_classes = config["n_classes"]

    out_overall_profile = profile.copy()
    out_overall_profile.update(
        {
            "dtype": "uint8",
            "compress": "LZW",
            "driver": "GTiff",
            "BIGTIFF": "YES",
            "tiled": True,
            "blockxsize": config["img_pixels_detection"],
            "blockysize": config["img_pixels_detection"],
        }
    )
    out_overall_profile["count"] = [2 if output_type == "argmax" else n_classes][0]
    # second band gives the max probability
    out = rasterio.open(path_out, "w+", **out_overall_profile)
    return out, path_out


# _________PIPELINES__________#
def run_from_config(config: dict) -> None:
    """Run the pipeline from a config file"""
    # setting up device and log
    device, use_gpu = setup_device(config)
    config = setup_out_path(config)

    run_pipeline(config, device, use_gpu)


def run_pipeline(config: dict, device: torch.device, use_gpu: bool) -> None:

    # extracting config parameters
    output_type = config["output_type"]
    n_classes = config["n_classes"]

    # log
    log_filename = os.path.join(
        config["local_out"],
        f"{config['output_name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    sys.stdout = Logger(filename=log_filename)
    sys.stderr = sys.stdout
    print(f"    [LOGGER] Writing logs to: {log_filename}")

    model = prepare_model(config, device)

    if config["compare"]:

        method_times = {}

        # common to a zone (= one tif image)
        metrics_json = config["local_out"] + "/metrics_per-patch_final.json"

        print(f"""    [ ] starting comparison...\n""")

        settings = gen_param_combination(config)
        for combi in settings:

            img_pixels_detection = combi["img_pixels_detection"]
            margin = combi["margin"]
            padding = combi["padding"]
            stride = combi["stride"]
            stitch = combi["stitching"]

            method = f"size={img_pixels_detection}_stride={stride}_margin={margin}_padding={padding}_stitching={stitch}"
            identifier = "_" + method

            # start timer
            start_time = datetime.datetime.now()

            dataset, data_loader, sliced_dataframe, profile, resolution = prepare_data(
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
                    device=device,
                    model=model,
                    use_gpu=use_gpu,
                    config=config,
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
                        resolution,
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

                    if config["metrics"]:
                        # compute metrics
                        inference_time = datetime.datetime.now() - start_time
                        inference_time = inference_time.total_seconds()
                        if method not in method_times:
                            method_times[method] = [inference_time]
                        else:
                            method_times[method].append(inference_time)

                        compute_metrics_patch(
                            prediction, window, config, method, metrics_json
                        )

            out.close()
            dataset.close_raster()  # type: ignore

            print(
                f"""    [X] done writing to {path_out.split('/')[-1]} raster file.\n"""
            )
            print(
                f"""    [X] done writing metrics to {metrics_json.split('/')[-1]} file.\n"""
            )

            if config["metrics"]:
                config["times"] = method_times

    else:

        # default configuration : exact clipping and default sized tiling

        stride = get_stride(config)[0]
        dataset, data_loader, sliced_dataframe, profile, resolution = prepare_data(
            config, stride
        )

        # prepare output raster
        out, path_out = prepare_output(config, profile)
        # inference loop

        print(f"""    [ ] starting inference...\n""")
        for samples in tqdm(data_loader):

            predictions, indices = inference(
                device=device,
                model=model,
                use_gpu=use_gpu,
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
                    resolution,
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


def batch_metrics_pipeline(config: dict, gt_dpt: str) -> None:
    """
    Compute metrics for a batch of images.
    Args:
        gt_dir (str): Path to the ground truth directory of the department.
        config (dict): Configuration, in which the parameters for the inference are specified
    """

    out_json = Path(config["metrics_out"])

    # output file
    assert out_json is not None, "Please provide an output path for the metrics"

    # __________INFERENCE__________#
    inputs_dpt = Path(config["input_path"])

    for full_zone in sorted(inputs_dpt.iterdir()):

        # find an input file image
        if not full_zone.is_dir():
            continue
        irc_path = next(full_zone.glob("*IRC.tif"), None)
        if irc_path is None:
            continue

        dpt, zone = str(irc_path).split("/")[-3:-1]
        gt_path = str(gt_dpt) + "/" + zone + "/"
        gt_path = next(Path(gt_path).glob("*.tif"), None)

        config["input_img_path"] = str(irc_path)
        config["truth_path"] = str(gt_path)
        config["output_name"] = str(irc_path).split("/")[-1].split(".")[0] + "-ARGMAX-S"

        # Inference and saving the predictions
        run_from_config(config)

    # we have all the predictions in the output folder

    out = out_json.with_suffix(".json")

    metrics_file = batch_metrics(config, gt_dpt)

    # save the metrics to a json file
    json.dump(
        metrics_file,
        open(out, "w"),
    )
    print(f"Metrics saved to {out}")


# __________Main function___________#
def main():

    # reading yaml
    args = argParser.parse_args()

    # setting up device and log
    config, device, use_gpu = setup(args)

    if args.batch_mode:
        gt_dir = config["truth_root"]
        gt_dpt = gt_dir + "/" + config["truth_path"].split("/")[-3]

        batch_metrics_pipeline(config, gt_dpt)
    else:
        run_pipeline(config, device, use_gpu)


if __name__ == "__main__":

    main()
