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
import re


def create_data(input_dir: str, truth_dir: str, data_type: str):
    input_dir_path = Path(input_dir)
    truth_dir_path = Path(truth_dir)

    input_files = [f for f in input_dir_path.rglob(f"*{data_type}.tif") if f.is_file()]
    truth_files = [f for f in truth_dir_path.rglob("*.tif") if f.is_file()]

    def extract_parts(stem):
        # assume stem is like "dpt_year_zone_modifier" or "dpt_year-zone-modifier"
        match = re.search(r"(\d{3})[_-]?(\d{4})[_-]?([A-Za-z0-9_]+)", stem)
        if match:
            return match.groups()
        return None

    # Build dictionaries keyed by (dept, year, zone)
    input_dict = {}
    for inp in input_files:
        parts = extract_parts(inp.stem)
        if parts:
            input_dict[parts] = inp
        else:
            print(f"Warning: Could not extract key from input file {inp.name}")

    truth_dict = {}
    for truth in truth_files:
        parts = extract_parts(truth.stem)
        if parts:
            truth_dict[parts] = truth
        else:
            print(f"Warning: Could not extract key from truth file {truth.name}")

    # Find intersection of keys
    common_keys = set(input_dict.keys()) & set(truth_dict.keys())

    data = []
    for key in tqdm(common_keys, desc="Creating dataset"):
        inp = input_dict[key]
        truth = truth_dict[key]
        data.append(
            {
                "input_img_path": os.path.join(input_dir, inp),
                "truth_path": os.path.join(truth_dir, truth),
            }
        )

    # Warn about unmatched files
    unmatched_inputs = set(input_dict.keys()) - common_keys
    unmatched_truths = set(truth_dict.keys()) - common_keys
    for key in unmatched_inputs:
        print(f"Warning: No matching label for input file {input_dict[key].name}")

    return pd.DataFrame(data)


def main():
    input_dir = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/ortho"
    truth_dir = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/labels_raster/FLAIR_19"
    data_type = "RVB"  # "IRC", "RVB", "RVBI", "RVBIE"

    df = create_data(input_dir, truth_dir, data_type)
    df.to_csv(f"data_paths_{data_type}.csv", index=False)

    print(f"Data paths saved to data_paths_{data_type}.csv")


if __name__ == "__main__":

    main()
