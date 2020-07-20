import numpy as np
import matplotlib.pyplot as plt
from .file_utils import makedir_if_not_exists

def make_image_grid_figure(images, fig_size=(10, 10), n_figures=25):
    n_figures = min(n_figures, 25)
    fig = plt.figure(figsize=fig_size)#, dpi=fig.dpi)
    sq = int(np.sqrt(n_figures))
    for i in range(n_figures):
        plt.subplot(sq,sq,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
    return fig

def save_plot_batch(images, file_name, fig_size=(10, 10), n_figures=25, make_dir=False):
    if make_dir:
        makedir_if_not_exists(file_name, is_path_dir=False)
    fig = make_image_grid_figure(images, fig_size=fig_size, n_figures=n_figures)
    fig.savefig(file_name)
    plt.close(fig)