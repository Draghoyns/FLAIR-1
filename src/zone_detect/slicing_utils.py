from typing import Any
import numpy as np
from shapely import Polygon
from shapely.geometry import box, mapping


def create_box_from_bounds(
    x_min: float, x_max: float, y_min: float, y_max: float
) -> Polygon:
    return box(x_min, y_max, x_max, y_min)


def create_polygon_from_bounds(
    x_min: float, x_max: float, y_min: float, y_max: float
) -> dict:
    return mapping(box(x_min, y_max, x_max, y_min))


def get_stride(config: dict[str, Any]) -> list[int]:
    img_size = config["img_pixels_detection"]

    ## handle default = no overlap handling
    if not config.get("overlap_strat", False):
        stride = [int(img_size - 2 * config["margin"])]
    else:  # overlap is handled and parameterized
        stride = [
            int(i * img_size) for i in config["strategies"]["tiling"]["stride_range"]
        ]
    return stride


def out_of_bounds(bigbox: list[float], box: list[float]) -> list[bool]:
    """Check if the coordinates are out of bounds"""

    oob = []
    left, right, bottom, top = bigbox
    for coord in box:
        if coord < left or coord > right or coord < bottom or coord > top:
            oob.append(True)
        else:
            oob.append(False)
    return oob


def get_tile_coord(
    start: int, end: int, limit: int, patch_size: int, stride: int
) -> list[int]:
    coords = []

    max_coord = limit - patch_size
    if max_coord < 0:
        return []

    tile_starts = set()
    for i in range(0, end, stride):
        if i + patch_size > limit:
            i = max_coord
        tile_starts.add(i)

    # Keep only tiles that intersect the [start, end) range
    for tile_start in tile_starts:
        tile_end = tile_start + patch_size
        if tile_end > start and tile_start < end:
            coords.append(tile_start)

    return coords


def nb_patches(
    img_size: tuple[int, int],
    stride: int,
) -> int:
    """
    Calculate the number of patches for a given image size.
    Lightweight version of slice_pixels that does not generate patches.
    """

    x_size, y_size = img_size

    # Calculate the number of patches in each dimension
    inexact_border_x = int(x_size % stride != 0)
    inexact_border_y = int(y_size % stride != 0)

    num_patches_x = x_size // stride + inexact_border_x
    num_patches_y = y_size // stride + inexact_border_y

    total_patches = num_patches_x * num_patches_y
    if inexact_border_x and inexact_border_y:
        # get rid of the last overlapping patch
        total_patches = total_patches - 1

    return total_patches


def patch_overlap(
    image_size: tuple[int, int],
    patch_size: int,
    query_bounds: list[int],
    stride: int,
) -> np.ndarray:
    """Works in pixels"""

    x_min, x_max, y_min, y_max = query_bounds
    overlap_map = np.zeros((patch_size, patch_size), dtype=np.uint8)

    image_size_x, image_size_y = image_size

    y_tiles = get_tile_coord(y_min, y_max, image_size_y, patch_size, stride)
    x_tiles = get_tile_coord(x_min, x_max, image_size_x, patch_size, stride)

    for tile_y in y_tiles:
        for tile_x in x_tiles:

            tile_y = min(tile_y, image_size_y - patch_size)
            tile_x = min(tile_x, image_size_x - patch_size)

            tile_ymax = tile_y + patch_size
            tile_xmax = tile_x + patch_size

            # Compute overlap between tile and the given patch
            inter_ymin = max(tile_y, y_min)
            inter_ymax = min(tile_ymax, y_max)
            inter_xmin = max(tile_x, x_min)
            inter_xmax = min(tile_xmax, x_max)

            if inter_ymax > inter_ymin and inter_xmax > inter_xmin:
                local_y_start = inter_ymin - y_min
                local_x_start = inter_xmin - x_min
                h = inter_ymax - inter_ymin
                w = inter_xmax - inter_xmin
                overlap_map[
                    local_y_start : local_y_start + h, local_x_start : local_x_start + w
                ] += 1

    # ensure no division by zero
    overlap_map[overlap_map == 0] = 1

    return overlap_map


def geo_patch_overlap(
    out: Any,  # DatasetWriter or similar object
    patch_size: int,
    query_bounds: list[float],
    big_box: list[float],
    stride: int,
) -> np.ndarray:
    """Works in geo coordinates.
    Converts geo coordinates to pixel coordinates and computes the overlap map, which represents how much overlap of patch there will be on a given pixel of the patch passed as argument.
    Args:
        out: DatasetWriter or similar object to get the profile and transform
        patch_size: size of the patch in pixels
        query_bounds: bounding box in geo coordinates [left, right, bottom, top]
        big_box: bounding box of the whole image in geo coordinates [left, right, bottom, top]
        stride: stride for the tiling
    Returns:
        overlap_map: numpy array representing the overlap map
    """
    transform = out.profile["transform"]
    size = out.profile["width"], out.profile["height"]

    # Convert geo coordinates to pixel coordinates
    query_bounds_pixel = geo_to_pixel(query_bounds, big_box, transform, patch_size)

    return patch_overlap(
        image_size=size,
        patch_size=patch_size,
        query_bounds=query_bounds_pixel,
        stride=stride,
    )


def patch_weights(patch_size: int, sigma: float, mode: str) -> np.ndarray:
    """Distance map to the center of the patch, given the patch"""
    center = patch_size // 2
    y, x = np.ogrid[:patch_size, :patch_size]
    dist = np.maximum(np.abs(y - center), np.abs(x - center))

    if mode == "gaussian":
        weights = np.exp(-dist / dist.max() ** 2) / (2 * sigma**2)
    else:
        weights = np.exp(-dist / dist.max() * sigma)  # smooth decay

    return weights


def total_weights(
    image_size: tuple[int, int],
    patch_size: int,
    query_bounds: list[int],
    stride: int,
    track_steps: bool = False,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Given the query, compute the total distance weights
    to which divide the average for each pixel"""

    steps = []

    x_min, x_max, y_min, y_max = query_bounds
    image_size_x, image_size_y = image_size

    map = np.zeros((y_max - y_min, x_max - x_min), dtype=np.float32)

    # we need tiles intersecting with the query
    y_tiles = get_tile_coord(y_min, y_max, image_size_y, patch_size, stride)
    x_tiles = get_tile_coord(x_min, x_max, image_size_x, patch_size, stride)

    # for each pixel in the query, if it is in a tile,
    # get the distance map of the tile and add the right value to the map

    weights = patch_weights(patch_size, sigma=0.5, mode="exp")

    for tile_y in y_tiles:
        for tile_x in x_tiles:

            # edge case
            tile_y = min(tile_y, image_size_y - patch_size)
            tile_x = min(tile_x, image_size_x - patch_size)

            # Compute overlap between tile and the given patch
            inter_ymin = max(tile_y, y_min)
            inter_ymax = min(tile_y + patch_size, y_max)
            inter_xmin = max(tile_x, x_min)
            inter_xmax = min(tile_x + patch_size, x_max)

            # if there is overlapping
            if inter_ymax > inter_ymin and inter_xmax > inter_xmin:
                local_y_start = inter_ymin - y_min
                local_x_start = inter_xmin - x_min
                local_y_tile_start = inter_ymin - tile_y
                local_x_tile_start = inter_xmin - tile_x
                h = inter_ymax - inter_ymin
                w = inter_xmax - inter_xmin
                map[
                    local_y_start : local_y_start + h,
                    local_x_start : local_x_start + w,
                ] += weights[
                    local_y_tile_start : local_y_tile_start + h,
                    local_x_tile_start : local_x_tile_start + w,
                ]
                if track_steps:
                    steps.append(map.copy())
    # no inversion
    # no normalization
    return map, steps


def geo_to_pixel(
    bounding_box: list[float],
    big_box: list[float],
    transform: list[float],
    img_size: int,
) -> list[int]:
    """Convert geo coordinates to pixel coordinates"""

    transformed = [
        int((bounding_box[0] - transform[2]) / transform[0]),
        int((bounding_box[1] - transform[5]) / transform[4]),
        int((bounding_box[2] - transform[2]) / transform[0]),
        int((bounding_box[3] - transform[5]) / transform[4]),
    ]
    x_min = min(transformed[0], transformed[2])
    y_min = min(transformed[1], transformed[3])

    pixelled_box = [
        x_min,
        x_min + img_size,
        y_min,
        y_min + img_size,
    ]

    # map limits to pixel coordinates
    if bounding_box[0] == big_box[0]:
        pixelled_box[0] = 0
    if bounding_box[1] == big_box[1]:
        pixelled_box[1] = img_size
    if bounding_box[2] == big_box[2]:
        pixelled_box[2] = 0
    if bounding_box[3] == big_box[3]:
        pixelled_box[3] = img_size

    return pixelled_box
