# This script computes the carbon emissions for a given dataset area
# It is designed fora run with fixed parameters, as it is intended to be used


import subprocess
from codecarbon import OfflineEmissionsTracker
import rasterio
from pathlib import Path


def compute_whole_area(dataset_dir: str) -> float:
    """
    Computes the total area of the dataset by summing the areas of all images.

    Args:
        dataset_dir (Path): Path to the dataset directory containing images.

    Returns:
        float: Total area in square meters.
    """

    total_area = 0.0
    images = Path(dataset_dir).rglob("*.tif")
    for image in images:
        if not image.is_file():
            continue

        with rasterio.open(image) as src:
            width = src.width  # in pixels
            height = src.height  # in pixels
            res_x, res_y = src.res  # pixel

        # Compute area
        pixel_area = abs(res_x * res_y)  # for 1 pixel in m²
        area = width * height * pixel_area
        total_area += area

    return total_area


if __name__ == "__main__":

    #### INPUT ####
    # ____________________________________________________#
    # dataset directory
    # config is supposed fixed, modifiable parameter but in default mode
    dataset_dir = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/ortho/D037_2021"

    config_path = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/inference_flair/config_detect_nonmodifiable.yaml"

    ######################################################

    #### INITIALIZATION ####
    # ____________________________________________________#
    # compute total area
    # still approx because area of 1 img != actual computed pixels
    area = compute_whole_area(dataset_dir)  # in m²
    print(f"Total area: {area} m2")

    tracker = OfflineEmissionsTracker(country_iso_code="FRA", measure_power_secs=1e9)
    tracker.start()

    ######################################################

    #### RUN CODE ####
    # ____________________________________________________#
    # run your code here

    command = "flair-detect --conf=" + config_path + " -b -c -m"
    print(f"Running command: {command}")
    arg = command.split()

    try:
        subprocess.run(arg, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during inference: {e}")

    ######################################################

    #### FINALIZATION ####
    # ____________________________________________________#
    emissions = tracker.stop()

    if emissions is None:
        print("No emissions data available.")
    else:

        scaled_area = area / 1e10  # scale area to 100km²
        scaled_emissions = emissions / scaled_area

        print(f"Raw emissions: {emissions} g CO2")
        print(f"Area: {scaled_area} 100km²")
        print(f"Emissions: {scaled_emissions} g CO2 / 100km²")
    ######################################################
