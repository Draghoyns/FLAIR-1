import os, sys
import datetime
import warnings

from geopandas import GeoDataFrame
from src.zone_detect.slicing_job import slice_extent
from src.zone_detect.test.metrics import compute_metrics
import torch
from tqdm import tqdm
import rasterio
import yaml
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from pytorch_lightning.utilities.rank_zero import rank_zero_only  # type: ignore

from src.zone_detect.model import load_model
from src.zone_detect.dataset import Sliced_Dataset
from src.zone_detect.test.tiles import get_stride
from src.zone_detect.compare import inference, stitching


warnings.simplefilter(action="ignore", category=FutureWarning)

#### CONF FILE
argParser = argparse.ArgumentParser()
argParser.add_argument("--conf", help="Path to the .yaml config file")
argParser.add_argument(
    "--compare", help="True to compare different methods", default=False, type=bool
)


# __________Logger___________#
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


# __________Config setup___________#


def read_config(file_path: str) -> dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def setup_device(config: dict) -> tuple[torch.device, bool]:
    """Setup the device"""

    use_gpu = False if torch.cuda.is_available() is False else config["use_gpu"]
    device = torch.device("cuda" if use_gpu else "cpu")

    return device, use_gpu


def setup_out_path(config: dict, args) -> tuple[dict, bool]:
    """Setup the output directory"""
    Path(config["output_path"]).mkdir(parents=True, exist_ok=True)

    if args.compare:
        # create a directory with a unique id
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        child_dir = os.path.join(config["output_path"], current_time)
        os.makedirs(child_dir, exist_ok=True)
        config["output_path"] = child_dir

    return config, args.compare


def setup(args) -> tuple[dict, torch.device, bool, bool]:
    """Setup the device and output path"""
    config = read_config(args.conf)
    device, use_gpu = setup_device(config)
    config, compare = setup_out_path(config, args)

    return config, device, use_gpu, compare


def setup_indiv_path(config: dict, identifier: str = "") -> tuple[dict, str]:

    assert isinstance(config["output_path"], str), "Output path does not exist."
    assert os.path.exists(config["input_img_path"]), "Input image path does not exist."
    assert (
        config["margin"] * 2 < config["img_pixels_detection"]
    ), "Margin is too large : margin*2 < img_pixels_detection"
    assert config["output_type"] in [
        "class_prob",
        "argmax",
    ], "Invalid output type: should be argmax or class_prob."
    assert config["norma_task"][0]["norm_type"] in [
        "custom",
        "scaling",
    ], "Invalid normalization type: should be custom or scaling."
    assert os.path.isfile(config["model_weights"]), "Model weights file does not exist."

    out_name = config["output_name"] + identifier

    if not out_name.endswith(".tif"):
        out_name += ".tif"

    try:
        # Path(config['output_path']).mkdir(parents=True, exist_ok=True)
        base_name = out_name
        path_out = os.path.join(config["output_path"], base_name)

        # Do not overwrite if file exists, add counter
        filename, ext = os.path.splitext(base_name)
        counter = 1

        while os.path.exists(path_out):
            new_name = f"{filename}_{counter}{ext}"
            path_out = os.path.join(config["output_path"], new_name)
            counter += 1
        # config['output_name'] = os.path.splitext(os.path.basename(path_out))[0]
        return config, path_out
    except Exception as error:
        print(f"Something went wrong during detection configuration: {error}")
        raise error  # avoid silent failure


# __________Logging___________#
# we're not handling multiple inputs yet
def conf_log(
    config: dict, resolution: tuple[float, float], img_size: list[int]
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
    
    |- overlapping strategy: {"handled" if compare_handling  else "exact"}
    |- tiling comparison: {"yes" if (compare_handling and config['strategies']['tiling']['enabled']) else "no"}
    |- stitching comparison: {"no" if not compare_handling else config['strategies']['stitching']['method']}
    |- padding: {"not handled" if not compare_handling else config["strategies"]['padding_overall']} \n
    """
    )


# __________Prepare objects___________#
def prepare_tiles(config: dict, stride: int) -> tuple[GeoDataFrame, dict, tuple]:
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
    conf_log(config, resolution, img_size)
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
            stitching_methods = ["exact_clipping"]

        # basically a grid comparison
        # could probably benefit from some optimization but ehhh
        print(f"""    [ ] starting comparison...\n""")
        for padding in padding_list:
            for img_pixels_detection in tile_size_list:
                config["img_pixels_detection"] = img_pixels_detection
                for margin in margin_list:
                    config["margin"] = margin
                    stride_list = get_stride(config)
                    for stride in stride_list:
                        for method in stitching_methods:

                            # slicing
                            sliced_dataframe, profile, resolution = prepare_tiles(
                                config, stride=stride
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

                            identifier = f"_size={img_pixels_detection}_stride={stride}_margin={margin}_padding={padding}_stitching={method}"
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
                                        method,
                                        stride,
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

                            out.close()
                            print(
                                f"""    [X] done writing to {path_out.split('/')[-1]} raster file.\n"""
                            )

    else:

        # default configuration : exact clipping and default sized tiling
        # slicing
        stride = get_stride(config)[0]
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
                    "exact_clipping",
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

    dataset.close_raster()

    if compare:
        compute_metrics(
            config,
        )

    sys.stdout = sys.__stdout__


if __name__ == "__main__":

    main()
