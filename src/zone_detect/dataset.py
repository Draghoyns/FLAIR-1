import os
import geopandas as gpd
import numpy as np

import torch
from torch.utils.data import Dataset

from skimage.util import img_as_float

from rasterio.enums import Resampling
import rasterio.windows


def convert(img: np.ndarray, img_type: str) -> np.ndarray:
    """Convert the image and keep the best class
    or keep all probabilities in separate bands and convert to color scale."""

    if img_type == "class_prob":
        if img.max() > 1:
            if np.issubdtype(img.dtype, np.integer):
                info = np.iinfo(img.dtype)
                img = img.astype(np.float32) / info.max  # normalize [0,1]
            else:
                img = (
                    img.astype(np.float32) / img.max()
                )  # fallback: normalize by max value
        img = (img * 255).astype(np.uint8)

    elif img_type == "argmax":
        img_arg = np.argmax(img, axis=0).astype(np.uint8)
        img_arg = np.expand_dims(img_arg, axis=0)

        img_max = np.max(img, axis=0).astype(np.float32)
        img_max = np.expand_dims(img_max, axis=0)
        img = np.concatenate([img_arg, img_max], axis=0)  # not sure about compatibility

    else:
        print("The output type has not been interpreted.")

    return img


# entry point is sound
def post_processing(
    output_type: str,
    path_before: str,
) -> None:
    """Post processing of the image if needed"""
    print(f"""    [ ] post processing...""")
    with rasterio.open(path_before) as src:
        img = src.read()
        profile = src.profile

    img = convert(img, output_type)
    profile["count"] = img.shape[0]
    os.remove(path_before)

    with rasterio.open(path_before, "w", **profile) as dst:
        dst.write(img)


class Sliced_Dataset(Dataset):

    def __init__(
        self,
        dataframe: gpd.GeoDataFrame,
        img_path: str,
        resolution: tuple[float, float],
        bands: list[int],
        patch_detection_size: int,
        norma_dict: list,
    ) -> None:

        self.dataframe = dataframe
        self.img_path = img_path
        self.resolution = resolution
        self.bands = bands
        self.num_bands = len(bands)
        self.height, self.width = patch_detection_size, patch_detection_size
        self.norma_dict = norma_dict[0]
        self.norm_type = self.norma_dict["norm_type"]
        self.norm_means = self.norma_dict["norm_means"]
        self.norm_stds = self.norma_dict["norm_stds"]
        self.big_image = rasterio.open(self.img_path)

    def __len__(self) -> int:
        return len(self.dataframe)

    def close_raster(self) -> None:
        if self.big_image and not self.big_image.closed:
            self.big_image.close()

    def normalization(self, in_img: np.ndarray) -> np.ndarray:
        if not self.norm_type in [
            "custom",
            "scaling",
        ]:
            print(
                "Invalid normalization type: should be custom or scaling. Going with scaling."
            )

        if self.norm_type == "custom":
            if len(self.norm_means) != len(self.norm_stds):
                print(
                    "If custom, provided normalization means and stds should be of the same length. Going with scaling."
                )
                return img_as_float(in_img)
            img = in_img.astype(np.float64)
            for i in range(self.num_bands):
                img[i] = (img[i] - self.norm_means[i]) / self.norm_stds[i]
            return img

        return img_as_float(in_img)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        try:
            bounds = self.dataframe.geometry.iat[index].bounds
            src = self.big_image

            window = rasterio.windows.from_bounds(
                *bounds, transform=src.meta["transform"]
            )
            patch_img = src.read(
                indexes=self.bands,
                window=window,
                out_shape=(self.num_bands, self.height, self.width),
                resampling=Resampling.bilinear,
                boundless=True,
            )

            patch_img = self.normalization(
                patch_img,
            )

            return {
                "image": torch.as_tensor(patch_img, dtype=torch.float),
                "index": torch.from_numpy(np.asarray([index])).int(),
            }

        except rasterio._err.CPLE_BaseError as error:  # type: ignore
            print(f"CPLE error {error}")
            return {
                "image": torch.zeros(
                    (self.num_bands, self.height, self.width), dtype=torch.float32
                ),
                "index": torch.tensor(index, dtype=torch.int32),
            }
