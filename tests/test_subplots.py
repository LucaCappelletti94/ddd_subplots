from ddd_subplots import subplots
import numpy as np
import os
from sklearn import datasets
from sklearn.decomposition import PCA


def test_subplots():
    X, y = datasets.load_iris(return_X_y=True)
    np.random.shuffle(X)
    X_reduced = PCA(n_components=3).fit_transform(X)
    colors = np.array(["red", "green", "blue"])[y]
    fig, axes = subplots(1, 3, figsize=(15, 5))
    for axis in axes.flatten():
        axis.scatter(*X_reduced.T, depthshade=False, c=colors, marker='o', s=20)
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.set_zticklabels([])

    fig.tight_layout()
    fig.savefig("test.png")