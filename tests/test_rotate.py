from ddd_subplots import subplots, rotate
import numpy as np
import os
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from numpy.linalg import norm


def my_func(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, *args, **kwargs):
    fig, axes = subplots(1, 3, figsize=(9, 3))
    axs = axes.flatten()
    axs[0].scatter(xs, ys, zs, **kwargs)
    axs[1].scatter(ys, zs, xs, **kwargs)
    axs[2].scatter(zs, xs, ys, **kwargs)
    for axis in axes.flatten():
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.set_zticklabels([])
    fig.tight_layout()
    return fig


def test_rotate():
    X, y = datasets.load_iris(return_X_y=True)
    X_reduced = PCA(n_components=3).fit_transform(X)
    colors = np.array(["red", "green", "blue"])[y]
    rotate(my_func, *X_reduced.T, path="test.gif",
           duration=2, fps=24, c=colors, marker='o', s=20)
