# Metrics Module

This module provides tools for evaluating and analyzing results using `metrics/main.py`.

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

## Arguments

- `--data`: Path to your dataset.csv (required)
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

- Python 3.x
- Any dependencies listed in `detect_requirements.txt` (if available).

## Notes

- Ensure the input dataset matches the expected structure.
- For more details, refer to the code in `main.py`.
- For custom parameters, you can manually change `frozen_config.yaml`

## TODO

- Add support for other types of model (ONNX, optimized version...)
- Add model loading from HuggingFace
