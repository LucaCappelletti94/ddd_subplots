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
    return np.real(np.exp(1j*theta)*w)/np.sqrt(2), np.imag(np.exp(1j*theta)*w)/np.sqrt(2), z


def _job(func: Callable, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, theta: float, args, kwargs, path):
    fig, axes = func(*rotate_z(xs, ys, zs, theta), *args, **kwargs)
    for axis in axes.flatten():
        axis.set_xticklabels([])
        axis.set_yticklabels([])
        axis.set_zticklabels([])
        axis.set_xlim(-1, 1)
        axis.set_ylim(-1, 1)
        axis.set_zlim(-1, 1)
    fig.savefig(path)
    plt.close(fig)


def _job_wrapper(task: Tuple):
    _job(*task)


def rotate(func: Callable, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, path: str, fps: int = 24, duration: int = 1, cache_directory: str = ".rotate", verbose: bool = False, *args, **kwargs):
    """Create rotating gif of given image.
        func: Callable, function return the figure.
        xs: np.ndarray, x coordinates.
        ys: np.ndarray, y coordinates.
        zs: np.ndarray, z coordinates.
        path: str, path where to save the GIF.
        fps: int = 24, number of FPS to create.
        duration: int = 1, duration of the rotation in seconds.
        cache_directory: str = ".rotate", directory where to store the frame.
        verbose: bool = False, whetever to be verbose about frame creation.
        *args, positional arguments to be passed to the `func` callable.
        **kwargs, keyword argument to be passed to the `func` callable
    """
    os.makedirs(cache_directory, exist_ok=True)
    xs/=xs.max()
    ys/=ys.max()
    zs/=zs.max() 
    tasks = [
        (
            func, xs, ys, zs, 2*np.pi * frame / (duration*fps), args, kwargs,
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
