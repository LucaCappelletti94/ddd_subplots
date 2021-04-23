from ddd_subplots import subplots, rotate
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from environments_utils import is_macos


def my_func(points: np.ndarray, *args, **kwargs):
    fig, axes = subplots(1, 3, figsize=(9, 3))
    axs = axes.flatten()
    axs[0].scatter(points[0], points[1], points[2], **kwargs)
    axs[1].scatter(points[1], points[2], points[0], **kwargs)
    axs[2].scatter(points[2], points[0], points[1], **kwargs)
    fig.tight_layout()
    return fig, axes


def test_rotate():
    X, y = datasets.load_iris(return_X_y=True)
    X_reduced = PCA(n_components=3).fit_transform(X)
    colors = np.array(["red", "green", "blue"])[y]
    rotate(my_func, X_reduced.T, path="test.gif",
           duration=3, fps=10, verbose=True, parallelize=False, c=colors, marker='o', s=20)
    if not is_macos():
        rotate(my_func, X_reduced.T, path="test.gif",
               duration=3, fps=10, c=colors, marker='o', s=20)
