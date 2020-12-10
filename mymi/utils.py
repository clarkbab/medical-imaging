from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

def plot_ct(*args):
    """
    input: required.
    label: optional.
    """
    # Get data.
    data = args[0]
    label = args[1] if len(args) >= 2 else None

    # Plot CT.
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(data), cmap='gray')
    
    # Plot label.
    if label is not None:
        colours = [(1.0, 1.0, 1.0, 0), (0.12, 0.47, 0.70, 1.0)]
        label_cmap = ListedColormap(colours)
        plt.imshow(np.transpose(label), cmap=label_cmap)

    plt.show()
