"""Create csv file with at least the structure:
    'input_img_path', 'truth_path'
from a list of image paths and a list of truth paths.

Assume paths look like:
    input_dir/possible_subdir../dpt_year_zone_IRC.tif
    truth_dir/dpt_year/zone/dpt_year-zone-MSK_FLAIR19-LABEL.tif

"""

from pathlib import Path
import pandas as pd
import os
from tqdm import tqdm


def create_data(input_dir, truth_dir):
    # Example implementation: match files by name in both directories

    input_dir_path = Path(input_dir)
    truth_dir_path = Path(truth_dir)

    input_files = sorted([f for f in input_dir_path.rglob("*IRC.tif") if f.is_file()])
    truth_files = sorted([f for f in truth_dir_path.rglob("*.tif") if f.is_file()])

    data = []
    for inp, truth in tqdm(zip(input_files, truth_files), total=len(input_files)):
        data.append(
            {
                "input_img_path": os.path.join(input_dir, inp),
                "truth_path": os.path.join(truth_dir, truth),
            }
        )
    return pd.DataFrame(data)


def main():
    input_dir = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/ortho"
    truth_dir = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/labels_raster/FLAIR_19"

    df = create_data(input_dir, truth_dir)
    df.to_csv("data_paths.csv", index=False)

    print("Data paths saved to data_paths.csv")


if __name__ == "__main__":

    main()
