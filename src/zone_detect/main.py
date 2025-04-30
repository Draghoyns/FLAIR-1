import os, sys
import datetime
import warnings

from geopandas import GeoDataFrame
from src.zone_detect.slicing_job import slice_extent, slice_extent_separate
import torch
from tqdm import tqdm
import rasterio
import argparse
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.rank_zero import rank_zero_only  # type: ignore

from src.zone_detect.model import load_model
from src.zone_detect.dataset import Sliced_Dataset
from src.zone_detect.test.metrics import compute_metrics_patch
from src.zone_detect.test.tiles import get_stride
from src.zone_detect.compare import inference, stitching

from src.zone_detect.utils import setup_device, setup_out_path, setup, setup_indiv_path


warnings.simplefilter(action="ignore", category=FutureWarning)

#### CONF FILE
argParser = argparse.ArgumentParser()
argParser.add_argument("--conf", help="Path to the .yaml config file")
argParser.add_argument(
    "--compare", help="True to compare different methods", default=False, type=bool
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
    compare_mode: bool,
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
    {compare_param if compare_mode else ""}
    """
    )


# __________Prepare objects___________#
def prepare_tiles(
    config: dict, stride: int, compare: bool
) -> tuple[GeoDataFrame, dict, tuple]:
    ## slicing extent for overlapping detection
    sliced_dataframe, profile, resolution, img_size = slice_extent(
        in_img=config["input_img_path"],
        patch_size=config["img_pixels_detection"],
        margin=config["margin"],
        output_name=config["output_name"],
        output_path=config["output_path"],
        write_dataframe=config["write_dataframe"],
        stride=stride,
    )
    ## log
    conf_log(config, resolution, img_size, compare)
    print(f"""    [x] sliced input raster to {len(sliced_dataframe)} squares...""")

    return sliced_dataframe, profile, resolution


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


def prepare_output_raster(
    path_out: str,
    profile: dict,
    img_pixels_detection: int,
    output_type: str,
    n_classes: int,
) -> rasterio.io.DatasetWriter:  # type: ignore
    """Prepare output raster profile"""
    out_overall_profile = profile.copy()
    out_overall_profile.update(
        {
            "dtype": "uint8",
            "compress": "LZW",
            "driver": "GTiff",
            "BIGTIFF": "YES",
            "tiled": True,
            "blockxsize": img_pixels_detection,
            "blockysize": img_pixels_detection,
        }
    )
    out_overall_profile["count"] = [2 if output_type == "argmax" else n_classes][0]
    # second band gives the max probability
    out = rasterio.open(path_out, "w+", **out_overall_profile)
    return out


# __________Main function___________#
def main():

    # reading yaml
    args = argParser.parse_args()

    # setting up device and log
    config, device, use_gpu, compare = setup(args)

    run_pipeline(config, device, use_gpu, compare)


def run_from_config(config: dict) -> None:
    """Run the pipeline from a config file"""
    # setting up device and log
    device, use_gpu = setup_device(config)
    config, compare = setup_out_path(config, None)

    run_pipeline(config, device, use_gpu, compare)


def run_pipeline(
    config: dict, device: torch.device, use_gpu: bool, compare: bool
) -> None:

    print("    [ ] no comparison" if not compare else "    [x] comparison")
    # extracting config parameters
    input_img_path = config["input_img_path"]
    channels = config["channels"]
    norma_task = config["norma_task"]
    batch_size = config["batch_size"]
    num_worker = config["num_worker"]
    output_type = config["output_type"]
    n_classes = config["n_classes"]

    img_pixels_detection = config["img_pixels_detection"]
    log_filename = os.path.join(
        config["output_path"],
        f"{config['output_name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )

    sys.stdout = Logger(filename=log_filename)
    sys.stderr = sys.stdout
    print(f"    [LOGGER] Writing logs to: {log_filename}")

    model = prepare_model(config, device)

    # initializing
    dataset = Sliced_Dataset(
        dataframe=GeoDataFrame(),
        img_path=input_img_path,
        resolution=tuple(),
        bands=channels,
        patch_detection_size=img_pixels_detection,
        norma_dict=norma_task,
    )
    if compare:
        # TODO
        padding_list = config["strategies"]["padding_overall"]
        if padding_list == []:
            padding_list = ["some-padding"]

        # configuration for comparison

        if config["strategies"]["tiling"]["enabled"]:
            tile_size_list = config["strategies"]["tiling"]["size_range"]
        else:
            tile_size_list = [img_pixels_detection]

        if config["strategies"]["stitching"]["enabled"]:
            margin_list = config["strategies"]["stitching"]["margin"]
            stitching_methods = config["strategies"]["stitching"]["method"]
        else:  # default stitching : exact clipping
            margin_list = [config["margin"]]
            stitching_methods = ["exact-clipping"]

        # basically a grid comparison
        # could probably benefit from some optimization but ehhh

        # common to a zone (= one tif image)
        metrics_json = config["output_path"] + "/metrics_per-patch_final.json"

        print(f"""    [ ] starting comparison...\n""")
        for padding in padding_list:
            for img_pixels_detection in tile_size_list:
                config["img_pixels_detection"] = img_pixels_detection
                for margin in margin_list:
                    config["margin"] = margin
                    # skip if parameters are not valid
                    if img_pixels_detection <= 2 * margin:
                        print(
                            f"""    [x] skipping {img_pixels_detection} pixels detection size with {margin} margin..."""
                        )
                        continue

                    stride_list = get_stride(config)
                    for stride in stride_list:
                        for stitch in stitching_methods:

                            # slicing
                            sliced_dataframe, profile, resolution = prepare_tiles(
                                config, stride, compare
                            )

                            # get dataset
                            dataset = Sliced_Dataset(
                                dataframe=sliced_dataframe,
                                img_path=input_img_path,
                                resolution=resolution,
                                bands=channels,
                                patch_detection_size=img_pixels_detection,
                                norma_dict=norma_task,
                            )

                            method = f"size={img_pixels_detection}_stride={stride}_margin={margin}_padding={padding}_stitching={stitch}"
                            identifier = "_" + method
                            config, path_out = setup_indiv_path(config, identifier)

                            # get Dataloader
                            data_loader = DataLoader(
                                dataset,
                                batch_size=batch_size,
                                num_workers=num_worker,
                                pin_memory=True,
                            )
                            # prepare output raster
                            out = prepare_output_raster(
                                path_out,
                                profile,
                                img_pixels_detection,
                                output_type,
                                n_classes,
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
                                # writing windowed raster to output rastert
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
                                        out.write_band(
                                            [1, 2], prediction, window=window
                                        )
                                    else:
                                        out.write_band(
                                            [i for i in range(1, n_classes + 1)],
                                            prediction,
                                            window=window,
                                        )

                                    compute_metrics_patch(
                                        prediction, window, config, method, metrics_json
                                    )

                            out.close()
                            print(
                                f"""    [X] done writing to {path_out.split('/')[-1]} raster file.\n"""
                            )
                            print(
                                f"""    [X] done writing metrics to {metrics_json.split('/')[-1]} file.\n"""
                            )

    else:

        # default configuration : exact clipping and default sized tiling
        # slicing
        stride = get_stride(config)[0]
        sliced_dataframe, profile, resolution = prepare_tiles(config, stride, compare)

        # get dataset
        dataset = Sliced_Dataset(
            dataframe=sliced_dataframe,
            img_path=input_img_path,
            resolution=resolution,
            bands=channels,
            patch_detection_size=img_pixels_detection,
            norma_dict=norma_task,
        )

        config, path_out = setup_indiv_path(config)

        # prepare output raster
        out = prepare_output_raster(
            path_out, profile, img_pixels_detection, output_type, n_classes
        )

        # get Dataloader
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_worker,
            pin_memory=True,
        )
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

    dataset.close_raster()

    sys.stdout = sys.__stdout__


if __name__ == "__main__":

    main()
