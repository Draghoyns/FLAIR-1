import numpy as np


def get_stride(config: dict) -> list:

    ## handle default = no overlap handling
    if "overlap_strat" not in config:
        stride = [config["img_pixels_detection"] - 2 * config["margin"]]
    elif config["overlap_strat"]:  # overlap is handled and parameterized
        stride = config["strategies"]["tiling"]["stride_range"]
    else:  # exact clipping
        stride = [config["img_pixels_detection"] - 2 * config["margin"]]
    return stride


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


def patch_overlap(
    image_size: tuple[int, int],
    patch_size: int,
    query_bounds: list[int],
    stride: int,
) -> np.ndarray:
    """Works in pixels"""

    x_min, x_max, y_min, y_max = query_bounds
    overlap_map = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)

    image_size_x, image_size_y = image_size

    y_tiles = get_tile_coord(y_min, y_max, image_size_y, patch_size, stride)
    x_tiles = get_tile_coord(x_min, x_max, image_size_x, patch_size, stride)

    for tile_y in y_tiles:
        for tile_x in x_tiles:

            if tile_y + patch_size > image_size_y:
                tile_y = image_size_y - patch_size
            if tile_x + patch_size > image_size_x:
                tile_x = image_size_x - patch_size

            tile_ymin = tile_y
            tile_ymax = tile_y + patch_size
            tile_xmin = tile_x
            tile_xmax = tile_x + patch_size

            # Compute overlap between tile and the given patch
            inter_ymin = max(tile_ymin, y_min)
            inter_ymax = min(tile_ymax, y_max)
            inter_xmin = max(tile_xmin, x_min)
            inter_xmax = min(tile_xmax, x_max)

            if inter_ymax > inter_ymin and inter_xmax > inter_xmin:
                local_y_start = inter_ymin - y_min
                local_x_start = inter_xmin - x_min
                h = inter_ymax - inter_ymin
                w = inter_xmax - inter_xmin
                overlap_map[
                    local_y_start : local_y_start + h, local_x_start : local_x_start + w
                ] += 1

    return overlap_map


def patch_weights(patch_size: int) -> np.ndarray:
    """Distance map to the center of the patch, given the patch"""
    dist = np.zeros((patch_size, patch_size))
    center = patch_size // 2
    for i in range(patch_size):
        for j in range(patch_size):
            dist[i, j] = max(abs(i - center), abs(j - center))

    # maybe use a gaussian kernel

    # smooth exponential decay

    sigma = 0.5
    dist = np.exp(-dist / dist.max() * sigma)  # smooth decay
    return dist


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

    for tile_y in y_tiles:
        for tile_x in x_tiles:

            # edge case
            if tile_y + patch_size > image_size_y:
                tile_y = image_size_y - patch_size
            if tile_x + patch_size > image_size_x:
                tile_x = image_size_x - patch_size

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
                weights = patch_weights(patch_size)
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
