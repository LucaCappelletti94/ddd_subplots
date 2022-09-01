from ddd_subplots import subplots, rotate
import numpy as np
import os
from sklearn import datasets
from sklearn.decomposition import PCA


def write_frame(X_reduced, y):
    colors = np.array(["red", "green", "blue"])[y]
    fig, axes = subplots(1, 2, figsize=(15, 5))
    for axis, X in zip(axes.flatten(), X_reduced):
        axis.scatter(*X.T, depthshade=False,
                     c=colors, marker='o', s=20)
    fig.tight_layout()
    return fig, axes


def test_rotate():
    X, y = datasets.load_iris(return_X_y=True)
    X_reduced = PCA(n_components=3).fit_transform(X)
    X_reduced_bis = 1 - X_reduced

    rotate(
        write_frame,
        [X_reduced, X_reduced_bis],
        "test_animation.gif",
        y,
        duration=10,
        verbose=True
    )
