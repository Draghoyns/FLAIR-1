# Metrics Module

This module provides tools for evaluating and analyzing results using `metrics/main.py`.

## Installation

The latest configuration that worked with all necessary dependencies was created as follows:
  ```bash
    conda create -n flair python=3.11
    conda activate flair

    cd FLAIR-1
    pip install -e .
    pip install -r src/zone_detect/metrics/requirements.txt
  ```

## Usage

1. **Prepare dataset:**

The script expects a csv file structured as follows:

    input_img_path, truth_path
    /input_dir/name_IRC.tif, /truth_dir/another_name.tif

If needed, `create_dataset_paths.py` provides a way to create the csv from the input and ground truth directories.


2. **Run the script:**
    ```bash
    python FLAIR-1/src/zone_detect/metrics/main.py --data=<path_to_csv> --ckpt=<path_to_ckpt>
    ```

Replace `<path_to_csv>` and `<path_to_ckpt>` with the actual paths.

If you decided to skip step 1 and don't give a `--data` argument, you will follow a step-by-step guide to create the dataset and then run the script.

## Arguments

- `--data`: Path to your dataset.csv
- `--ckpt`: Path to the model chekpoints (required) (only support swin-upernet for now)

- Additional options may be available; run:
  ```bash
  python main.py --help
  ```
  to see all supported arguments.

## Example

```bash
python main.py --data=data_paths.csv --ckpt=SwinUpernet_Small.ckpt
```

## Requirements

- Python 3.x (3.11 is advised, no guarantees on dependencies if you use other versions)
- Any dependencies listed in `requirements.txt` (if available).
- some conda packages are needed too :
```bash
conda install -c nvidia cuda-toolkit
conda install -c conda-forge cudnn
```
-> refer to `environment.yaml`

## Notes

- Ensure the input dataset matches the expected structure. You can always create a new one with by providing no `--data` in the command.
- For more details, refer to the code in `main.py`.
- For custom parameters, you can manually change `frozen_config_***.yaml`
- If multiple options are provided as a list in the config parameters (e.g. `img_size_detection`), the script will automatically run for all possible combination of parameters ad stack the related metrics per combination.

## TODO

- Add support for other types of model (ONNX, optimized version...)
- Add model loading from HuggingFace
