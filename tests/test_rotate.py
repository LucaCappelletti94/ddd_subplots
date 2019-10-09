from ddd_subplots import subplots, rotate
import numpy as np
import os
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from numpy.linalg import norm



def my_func(xs, ys, zs, *args, **kwargs):
    fig, axes = subplots(1, 3, figsize=(9, 3))
    X = MinMaxScaler().fit_transform(np.array([xs, ys, zs]).T)
    distances = 1-norm(X-np.array([1,1,1]), axis=1, ord=2)
    for axis in axes.flatten():
        axis.scatter(*X.T, **kwargs, zorder=distances)
    fig.tight_layout()
    return fig


def test_rotate():
    X, y = datasets.load_iris(return_X_y=True)
    X_reduced = PCA(n_components=3).fit_transform(X)
    colors = np.array(["red", "green", "blue"])[y]
    rotate(my_func, *X_reduced.T, path="test.gif", duration=2, fps=24, depthshade=False, c=colors, marker='o', s=20)
    os.remove("test.gif")