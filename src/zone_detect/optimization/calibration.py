import random
from pathlib import Path
import numpy as np
import rasterio
import torch
from tqdm import tqdm

from src.zone_detect.utils import read_config
from src.zone_detect.prepare import prepare_data

from rasterio.enums import Resampling


config_path = "./configs/20250804_config_detect_latest-update.yaml"

dataset_size = 200

input_dir_path = "/var/tmp/shys/INFERENCE_HS/DATA/dataset_zone_last/ortho/D037_2021"
save_path = "./0testing_saves/calibration_dataset"

# random seed for reproducibility
# randomly select 200 images from the input directory (with replacement)"
# map reduce to get image path : count
# for each image path, tile the image
# get count number of patches
# save each patch to the output directory with a unique name


# random seed
seed = 42
random.seed(seed)


def create_calib_dataset(config_path: str, dataset_size: int, save_path) -> str:
    config = read_config({"conf": config_path})

    config["batch_size"] = 1  # Set batch size to 1 for calibration dataset

    input_dir = Path(input_dir_path)
    output_dir = Path(save_path)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_dir.rglob("*IRC.tif"))
    selected_images = random.choices(image_paths, k=dataset_size)

    print(
        f"Selected {len(selected_images)} images for calibration dataset (with replacement). Reminder: wanted {dataset_size}."
    )

    image_count = dict()
    for image_path in selected_images:
        image_count[image_path] = image_count.get(image_path, 0) + 1

    next_idx = 0

    for image_path in tqdm(image_count.keys(), desc="Processing images"):
        config["input_img_path"] = str(image_path)

        # slice image
        _, data_loader, _, _ = prepare_data(config)  # type: ignore
        nb_draws = image_count[image_path]

        # draw patches
        reservoir = get_k_patches(data_loader, k=nb_draws, seed=seed)

        # save patches
        next_idx = save_calibration_images(reservoir, save_path, start_idx=next_idx)

    print(f"Calibration dataset created at {save_path}")
    return save_path


def get_k_patches(dataloader, k: int, seed=None) -> list[dict[str, torch.Tensor]]:
    """
    Randomly sample k samples from a dataloader using reservoir sampling.

    Args:
        dataloader: The dataloader to sample from
        k: Number of samples to select
        seed: Random seed for reproducibility (optional)

    Returns:
        List of k randomly selected samples, each containing a single 'image' key
    """
    if seed is not None:
        random.seed(seed)

    reservoir = []

    for i, samples in enumerate(dataloader):
        # Handle batch case - if samples contain batches, iterate through batch

        if (
            isinstance(samples["image"], torch.Tensor) and samples["image"].dim() > 3
        ):  # Assuming images are batched
            batch_size = samples["image"].shape[0]
            for j in range(batch_size):
                sample = {
                    "image": samples["image"][j],
                }

                current_idx = i * batch_size + j

                if len(reservoir) < k:
                    reservoir.append(sample)
                else:
                    # Reservoir sampling: replace with probability k/current_idx
                    replace_idx = random.randint(0, current_idx)
                    if replace_idx < k:
                        reservoir[replace_idx] = sample
        else:
            # Handle single sample case
            sample = {
                "image": (
                    samples["image"].squeeze(0)
                    if samples["image"].dim() == 4
                    else samples["image"]
                ),
            }

            if len(reservoir) < k:
                reservoir.append(sample)
            else:
                # Reservoir sampling: replace with probability k/i
                replace_idx = random.randint(0, i)
                if replace_idx < k:
                    reservoir[replace_idx] = sample

    return reservoir


def tensor_to_rasterio_array(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a tensor to numpy array suitable for rasterio.

    Args:
        tensor: Image tensor of shape (C, H, W) with values in [0, 1] or [-1, 1]

    Returns:
        Numpy array of shape (C, H, W) with appropriate dtype
    """
    # Handle different tensor formats
    if tensor.dim() == 4:  # Remove batch dimension if present
        tensor = tensor.squeeze(0)

    # Ensure tensor is (C, H, W)
    if tensor.dim() == 2:  # Add channel dimension for grayscale
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() == 3 and tensor.shape[0] > tensor.shape[2]:  # Likely (H, W, C)
        tensor = tensor.permute(2, 0, 1)

    # Convert to numpy
    array = tensor.detach().cpu().numpy()

    # Normalize to [0, 1] if needed
    if array.min() < 0:  # Assume [-1, 1] range
        array = (array + 1) / 2

    # Convert to appropriate dtype
    # Use float32 to preserve precision, or uint16 for better range
    if array.max() <= 1.0:
        array = (array * 65535).astype(np.uint16)  # Use full uint16 range
    else:
        array = array.astype(np.float32)

    return array


def save_calibration_images(
    samples: list[dict[str, torch.Tensor]],
    save_dir: str,
    start_idx: int = 0,
    compress: str = "lzw",
) -> int:
    """
    Save calibration samples as TIFF files using rasterio.

    Args:
        samples: List of samples from create_calibration_dataset
        save_dir: Directory to save images
        compress: Compression method ('lzw', 'deflate', 'jpeg', 'none')
        start_idx: Starting index for naming files
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # print(f"Saving {len(samples)} calibration images to {save_dir}")

    for i, sample in enumerate(samples):
        image_tensor = sample["image"]

        array = tensor_to_rasterio_array(image_tensor)
        channels, height, width = array.shape

        image_file = save_path / f"calibration_{start_idx + i:04d}.tif"
        with rasterio.open(
            image_file,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=channels,
            dtype=array.dtype,
            compress=compress,
            tiled=True,  # Enable tiling for better performance
            blockxsize=512,
            blockysize=512,
        ) as dst:
            # Write each channel
            for channel in range(channels):
                dst.write(array[channel], channel + 1)

    # print(f"Saved {len(samples)} images successfully")
    return start_idx + len(samples)


####__________LOAD DATASET__________####


def load_calibration_images(
    save_dir: str,
    target_size: tuple[int, int] = (512, 512),
    normalize: bool = True,
    to_float: bool = True,
) -> list[torch.Tensor]:
    """
    Load calibration images from directory using rasterio.

    Args:
        save_dir: Directory containing saved TIFF images
        target_size: Optional (height, width) to resize images
        normalize: Whether to normalize to [0, 1] range or apply ImageNet stats if RGB
        to_float: Whether to convert to float32

    Returns:
        List of image tensors
    """
    save_path = Path(save_dir)
    tiff_files = sorted(list(save_path.glob("*.tif")) + list(save_path.glob("*.tiff")))

    if not tiff_files:
        raise ValueError(f"No TIFF files found in {save_dir}")

    images = []
    for tiff_file in tiff_files:
        with rasterio.open(tiff_file) as src:
            # Read all bands
            array = src.read()  # (C, H, W)

            if to_float:
                if array.dtype == np.uint16:
                    array = array.astype(np.float32) / 65535.0
                elif array.dtype == np.uint8:
                    array = array.astype(np.float32) / 255.0
                else:
                    array = array.astype(np.float32)

            if target_size:
                target_h, target_w = target_size
                resized = np.zeros(
                    (array.shape[0], target_h, target_w), dtype=array.dtype
                )
                for i in range(array.shape[0]):
                    resized[i] = src.read(
                        i + 1,
                        out_shape=(target_h, target_w),
                        resampling=Resampling.bilinear,
                    )
                array = resized

        tensor = torch.from_numpy(array)

        if normalize and tensor.max() > 1.0:
            tensor = tensor / tensor.max()

        if normalize and tensor.shape[0] == 3:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = (tensor - mean) / std

        images.append(tensor.unsqueeze(0))  # Add batch dimension

    batch_size = 40
    # Create batches of samples with batch_size along the first axis
    batched_samples_list = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        # Ensure all samples are on the same device and have the same shape
        batch = [s.to("cuda") for s in batch]
        batch_tensor = torch.cat(batch, dim=0)
        batched_samples_list.append(batch_tensor)

    print(f"Loaded {len(images)} calibration images from {save_dir}")
    return batched_samples_list


if __name__ == "__main__":
    out = create_calib_dataset(config_path, dataset_size, save_path)

    load_calibration_images(out)
