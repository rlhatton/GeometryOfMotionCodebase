import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

crimson = [234/255, 14/255, 30/255]

negative_colors = np.linspace([0, 0, 0], [1, 1, 1])
positive_colors = np.linspace([1, 1, 1], crimson)

crimson_cmp = ListedColormap(np.concatenate((negative_colors, positive_colors)))
