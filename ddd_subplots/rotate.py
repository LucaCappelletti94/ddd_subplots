from matplotlib.axis import Axis
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Tuple
from tqdm.auto import tqdm
import os
from sklearn.preprocessing import MinMaxScaler
import imageio
from pygifsicle import optimize
import shutil


def rotate_z(x: np.ndarray, y: np.ndarray, z: np.ndarray, theta: float):
    w = x+1j*y
    return MinMaxScaler().fit_transform(np.array([
        np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z
    ]).T).T


def _job(func: Callable, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, theta: float, args, kwargs, path):
    fig = func(*rotate_z(xs, ys, zs, theta), *args, **kwargs)
    fig.savefig(path)
    plt.close(fig)


def _job_wrapper(task: Tuple):
    _job(*task)


def rotate(func: Callable, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, path: str, fps: int = 24, duration: int = 1, cache_directory: str = ".rotate", verbose: bool = False, *args, **kwargs):
    os.makedirs(cache_directory, exist_ok=True)
    X = MinMaxScaler().fit_transform(np.array([xs, ys, zs]).T).T
    tasks = [
        (
            func, *X, 2*np.pi * frame / (duration*fps), args, kwargs,
            "{cache_directory}/{frame}.jpg".format(
                cache_directory=cache_directory,
                frame=frame
            )
        )
        for frame in range(duration*fps)
    ]

    for task in tqdm(tasks, desc="Rendering frames", disable=not verbose):
        _job_wrapper(task)

    with imageio.get_writer(path, mode='I', fps=fps) as writer:
        for task in tqdm(tasks, disable=not verbose):
            writer.append_data(imageio.imread(task[-1]))

    optimize(path)
    shutil.rmtree(cache_directory)
