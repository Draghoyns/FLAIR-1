import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def viz_slicing(img_size: tuple[int, int], patches: set) -> None:

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
