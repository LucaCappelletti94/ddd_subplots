from ddd_subplots import subplots, rotate
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA


def my_func(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, *args, **kwargs):
    fig, axes = subplots(1, 3, figsize=(9, 3))
    axs = axes.flatten()
    axs[0].scatter(xs, ys, zs, **kwargs)
    axs[1].scatter(ys, zs, xs, **kwargs)
    axs[2].scatter(zs, xs, ys, **kwargs)
    fig.tight_layout()
    return fig, axes


def test_rotate():
    X, y = datasets.load_iris(return_X_y=True)
    X_reduced = PCA(n_components=3).fit_transform(X)
    colors = np.array(["red", "green", "blue"])[y]
    rotate(my_func, *X_reduced.T, path="test.gif",
           duration=2, fps=24, c=colors, marker='o', s=20)
