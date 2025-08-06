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

    print(f"Found {len(input_files)} input files and {len(truth_files)} truth files.")

    def extract_parts(stem):
        # assume stem is like "dpt_year_zone_modifier" or "dpt_year-zone-modifier"
        # zone: 2 letters + underscore + letters/numbers/underscores ending with number
        # identifier: starts with letter + letters/numbers/underscores
        match = re.search(
            r"(\d{3})[_-]?(\d{4})[_-]?([A-Z]{2}_[A-Za-z0-9_]*\d)[_-]([A-Za-z][A-Za-z0-9_]*)",
            stem,
        )
        if match:
            dept, year, zone, identifier = match.groups()
            print(
                f"Extracted parts from {stem}: dept={dept}, year={year}, zone={zone}, identifier={identifier}"
            )
            return (dept, year, zone)  # Only return dept, year, zone for matching
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


def interactive_dataset() -> str:
    """Interactive function to get dataset paths from user input."""

    print("\nYou did not provide a dataset paths file, let's create one together !\n")

    input_dir = input("\nWhat is the input data main folder ? ")
    input_dir = input_dir.strip()
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory {input_dir} does not exist.")

    truth_dir = input("\nWhat is the ground truth main folder ? ")
    truth_dir = truth_dir.strip()
    if not os.path.isdir(truth_dir):
        raise ValueError(f"Truth directory {truth_dir} does not exist.")

    data_type = input("\nWhat is the data type (IRC, RVB, RVBI, RVBIE): ")
    data_type = data_type.strip().upper()
    if data_type not in ["IRC", "RVB", "RVBI", "RVBIE"]:
        raise ValueError(
            f"Invalid data type {data_type}. Must be one of: IRC, RVB, RVBI, RVBIE."
        )
    print(f"\nCreating dataset for {data_type} data...")
    df = create_data(input_dir, truth_dir, data_type)
    df.to_csv(f"data_paths_{data_type}.csv", index=False)
    print(f"Data paths saved to data_paths_{data_type}.csv\n")

    return f"data_paths_{data_type}.csv"


def main():
    input_dir = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/ortho"
    truth_dir = "/media/DATA/INFERENCE_HS/DATA/dataset_zone_last/labels_raster/FLAIR_19"
    data_type = "RVB"  # "IRC", "RVB", "RVBI", "RVBIE"

    df = create_data(input_dir, truth_dir, data_type)
    df.to_csv(f"data_paths_{data_type}.csv", index=False)

    print(f"Data paths saved to data_paths_{data_type}.csv")


if __name__ == "__main__":

    main()
