def slice_pixels(
    img_size: tuple[int, int],
    patch_size: int,
    margin: int,
    stride: int,
) -> list[tuple[int, int, int, int]]:
    """
    Generate patches for a given image size.
    The patches are the small boxes where the margins were removed.
    They will be added for inference inside the slice_geo function."""

    def _add_patch_if_valid(patches, x_min, y_min):
        x_max = x_min + patch_size
        y_max = y_min + patch_size
        if x_max <= x_size and y_max <= y_size:
            patches.add((x_min, x_max, y_min, y_max))
        return patches

    patches = set()
    x_size, y_size = img_size

    patch_size = patch_size - 2 * margin

    for y in range(0, y_size + 1, stride):
        for x in range(0, x_size + 1, stride):
            # bottom right corner
            patches = _add_patch_if_valid(patches, x, y)

    # add edge cases
    if y_size - patch_size > 0 and (y_size - patch_size) % stride != 0:
        # bottom
        y = y_size - patch_size
        for x in range(0, x_size - patch_size + 1, stride):
            patches = _add_patch_if_valid(patches, x, y)

    if x_size - patch_size > 0 and (x_size - patch_size) % stride != 0:
        # right
        x = x_size - patch_size
        for y in range(0, y_size - patch_size + 1, stride):
            patches = _add_patch_if_valid(patches, x, y)

        # add the last patch
    if (
        y_size - patch_size > 0
        and (y_size - patch_size) % stride != 0
        and x_size - patch_size > 0
        and (x_size - patch_size) % stride != 0
    ):
        y = y_size - patch_size
        x = x_size - patch_size
        patches = _add_patch_if_valid(patches, x, y)

    return sorted(patches)
