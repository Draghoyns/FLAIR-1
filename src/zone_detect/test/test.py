import matplotlib.pyplot as plt
from src.zone_detect.test.tiles import patch_weights


# example

if __name__ == "__main__":
    img_size = 5, 5
    patch_size = 3
    margin = 0
    stride = 2
    query = (0, 5, 0, 5)

    local_weights = patch_weights(patch_size)

    # visualize

    plt.plot(local_weights)
    plt.title("Weights")
    plt.xlabel("Patch index")
    plt.ylabel("Weight")
    plt.show()
