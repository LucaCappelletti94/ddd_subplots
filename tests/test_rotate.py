from ddd_subplots import subplots, rotate
import numpy as np
from sklearn import datasets
from sklearn.decomposition import PCA
from environments_utils import is_macos


def my_func(points: np.ndarray, *args, **kwargs):
    fig, axis = subplots(1, 1, figsize=(5, 5), dpi=100)
    axis.scatter(*points, **kwargs)
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_zticklabels([])
    axis.set_xlim(-0.3, 0.3)
    axis.set_ylim(-0.3, 0.3)
    axis.set_zlim(-0.3, 0.3)
    axis.set_axis_off()
    fig.tight_layout()
    return fig, axis


def test_rotate():
    X, y = datasets.load_iris(return_X_y=True)
    X_reduced = PCA(n_components=3).fit_transform(X)
    colors = np.array(["red", "green", "blue"])[y]
    rotate(
        my_func,
        X_reduced.T,
        path="test.gif",
        duration=10,
        fps=30,
        verbose=True,
        parallelize=False,
        c=colors,
        marker='o',
        s=10
    )
    rotate(
        my_func,
        X_reduced.T,
        path="test.mp4",
        duration=10,
        fps=30,
        verbose=True,
        parallelize=False,
        c=colors,
        marker='o',
        s=10
    )
    if not is_macos():
        rotate(
            my_func,
            X_reduced.T,
            path="test.gif",
            duration=10,
            fps=30,
            c=colors,
            marker='o',
            s=10
        )
