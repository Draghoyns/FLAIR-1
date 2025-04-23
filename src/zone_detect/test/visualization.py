import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
import numpy as np

from src.zone_detect.test.pixel_operation import slice_pixels
from src.zone_detect.test.tiles import total_weights


def viz_slicing(img_size: tuple[int, int], patches: set | list) -> None:

    x_size, y_size = img_size
    _, ax = plt.subplots()
    ax.set_xlim(0, x_size)
    ax.set_ylim(0, y_size)
    ax.set_aspect("equal")

    ax.invert_yaxis()

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # Draw the patches
    # if it's the first patch, draw in a different color
    for patch in patches:
        x_min, x_max, y_min, y_max = patch
        if x_min == 0 and y_min == 0:
            # Green dashed rectangle with double linewidth
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=2,
                edgecolor="g",
                facecolor="none",
                linestyle="--",
            )
        else:
            # Red semi-transparent rectangle
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
                alpha=0.2,
            )

        ax.add_patch(rect)

    plt.xlim(0, x_size)
    plt.ylim(y_size, 0)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()


def visualize_total_weights_steps(steps: list[np.ndarray], patch_size: int):
    cmap = plt.get_cmap("nipy_spectral")
    norm = Normalize(vmin=0, vmax=6.5)

    fig, ax = plt.subplots()

    im = ax.imshow(steps[0], cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax)
    ax.set_title("Tile Step 0")

    ax.set_xlim(0, patch_size)
    ax.set_ylim(patch_size, 0)

    def update(i):
        im.set_data(steps[i])
        ax.set_title(f"Tile Step {i}")
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal step
        if event.key == "right":
            step = (step + 1) % len(steps)
            update(step)
        elif event.key == "left":
            step = (step - 1) % len(steps)
            update(step)

    step = 0
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


if __name__ == "__main__":
    img_size = 10, 10
    patch_size = 3
    margin = 0
    stride = 2
    query = [0, 10, 0, 10]

    patches = slice_pixels(
        img_size=img_size,
        patch_size=patch_size,
        margin=margin,
        stride=stride,
    )

    viz_slicing(img_size, patches)

    map, steps = total_weights(
        img_size,
        patch_size,
        query,
        stride,
        track_steps=True,
    )

    # visualize the steps
    visualize_total_weights_steps(steps, patch_size)
