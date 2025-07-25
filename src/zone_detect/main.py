import argparse
import datetime
import json
import sys
from tqdm import tqdm
import warnings

from pathlib import Path
from typing import Any

from src.zone_detect.dataset import post_processing
from src.zone_detect.inference import inference
from src.zone_detect.prepare import (
    prepare_model,
    prepare_output,
    prepare_data,
)
from src.zone_detect.stitching_job import stitching

from src.zone_detect.metrics.metrics import (
    batch_metrics,
    compute_metrics_patch,
)

from src.zone_detect.utils import (
    batchmode_path_setup,
    gen_param_combination,
    open_images,
    setup_device,
    setup_out_path,
    setup,
    Logger,
    timer,
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


# _________PIPELINES__________#
def run_from_config(config: Config) -> None:
    """Run the pipeline from a config file"""
    # setting up device and log
    # TODO: add setup operations to the config
    device, _ = setup_device(config)

    run_pipeline(config)


def run_pipeline(config: Config) -> None:
    """Works for a single input image"""

    # set up common output path
    config = setup_out_path(config)

    # extracting config parameters
    output_type = config["output_type"]
    n_classes = config["n_classes"]
    compare = config["compare"]
    local_out = Path(config["local_out"])
    compute_metrics = config["metrics"]
    model_type = config.get("model_type", "pytorch")
    model_args = config.get("model_args", dict())

    processed_area = config.get("processed_area", 0.0)

    # log
    log_filename = local_out / Path(
        f"{config['output_name']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )
    sys.stdout = Logger(filename=str(log_filename))
    sys.stderr = sys.stdout
    print(f"    [LOGGER] Writing logs to: {log_filename}")

    # setup elements for the metrics
    truth_array, metrics_json = open_images(
        config,
        local_out,
        compute_metrics,
    )

    method_times = config.get("times", dict())
    method_patches = config.get("nb_patches", dict())

    settings = gen_param_combination(config, compare)

    for combi in settings:

        method_metrics_per_patch = []
        timings = {}

        img_pixels_detection = combi["img_pixels_detection"]
        margin = combi["margin"]
        padding = combi["padding"]
        stride = combi["stride"]
        stitch = combi["stitching"]

        effective_output_type = (
            output_type if stitch == "exact-clipping" else "class_prob"
        )

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

        method_start = datetime.datetime.now()

        timed_prepare_data = timer(timings)(prepare_data)
        # timer_data = datetime.datetime.now()

        dataset, data_loader, sliced_dataframe, profile = timed_prepare_data(config)
        # data_prep_time = ( datetime.datetime.now() - timer_data).total_seconds() * 1000  # ms
        data_prep_time = timings["prepare_data"]

        single_area = sliced_dataframe["geometry"].area.sum()

        # prepare output raster
        out, path_out = prepare_output(
            config,
            profile,
            identifier,
        )

        pure_infer_time = 0  # ms
        data_write_time = 0  # ms

        timed_inference = timer(timings)(inference)

        print(f"""    [ ] starting inference...\n""")
        for samples in tqdm(data_loader):

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
                    combi,
                    sliced_dataframe,
                    prediction,
                    index,
                    out,
                    config["effective_output_type"],
                )

                prediction_to_write = prediction.copy()
                prediction_to_write[1:] = prediction[1:] * 65535
                prediction_to_write = prediction_to_write.astype("uint16")
                # add threshold to post process probabilities -> e.g. range [0.5, 0.9]

                # write
                if effective_output_type == "argmax":
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

                if compute_metrics:
                    # compute metrics per patch

                    method_metrics_per_patch.append(
                        compute_metrics_patch(
                            prediction,
                            truth_array,
                            window,
                            config,
                            method,
                        )
                    )
        # end of loop on one method
        ### timing
        total_time = (
            datetime.datetime.now() - method_start
        ).total_seconds() * 1000  # ms
        if method not in method_times:
            method_times[method] = {
                "data_prep_time": [data_prep_time],
                "pure_infer_time": [pure_infer_time],
                "data_write_time": [data_write_time],
                "total_time": [total_time],
            }
        else:
            method_times[method]["data_prep_time"].append(data_prep_time)
            method_times[method]["pure_infer_time"].append(pure_infer_time)
            method_times[method]["data_write_time"].append(data_write_time)
            method_times[method]["total_time"].append(total_time)

        ### patches
        if method not in method_patches:
            method_patches[method] = [len(sliced_dataframe)]
        else:
            method_patches[method].append(len(sliced_dataframe))

        out.close()
        dataset.close_raster()  # type: ignore

        print(f"""    [X] done writing to {path_out.split('/')[-1]} raster file.\n""")

        if compute_metrics:
            config["times"] = method_times
            config["nb_patches"] = method_patches

            # per patch metrics
            with open(metrics_json, "w") as f:
                json.dump(method_metrics_per_patch, f, indent=2)

            print(f"""    [X] done writing metrics to {metrics_json.name} file.\n""")
            processed_area += single_area

        if effective_output_type != output_type:
            post_processing(output_type, path_out)

    sys.stdout = sys.__stdout__

    config.update({"processed_area": processed_area})


def batch_metrics_pipeline(config: Config, truth_dpt: Path) -> None:
    """
    Compute metrics for a batch of images.
    Args:
        gt_dpt (Path): Path to the ground truth directory of the department.
        config (dict): Configuration, in which the parameters for the inference are specified
    """

    out_json = config.get("metrics_out", "")
    data_type = config["data_type"]  # IRC, RVB etc.
    file_pattern = f"*{data_type}.tif"
    compute_metrics = config["metrics"]

    # __________INFERENCE__________#
    inputs_dpt = Path(config["input_path"])

    zone_list = sorted(p for p in inputs_dpt.iterdir() if p.is_dir())
    for full_zone in zone_list:

        # find an input file image
        img_path = next(full_zone.glob(file_pattern), None)
        if img_path is None:
            continue

        # set up config for the current image
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
        run_pipeline(config)

    # we have all the predictions in the output folder

    if compute_metrics:

        out = Path(out_json).with_suffix(".json")
        metrics_file = batch_metrics(config, truth_dpt)

        # save the metrics to a json file

        with open(out, "w") as f:
            json.dump(metrics_file, f, indent=2)

        print(f"Metrics saved to {out}")


# __________Main function___________#
def main():

    args = argParser.parse_args().__dict__

    # setting up device and log
    config, device, use_gpu = setup(args)

    # model
    config = prepare_model(config, device)

    if args["batch_mode"]:
        config, gt_dpt = batchmode_path_setup(config, use_gpu)

        batch_metrics_pipeline(config, gt_dpt)
    else:
        run_pipeline(config)


if __name__ == "__main__":

    main()
