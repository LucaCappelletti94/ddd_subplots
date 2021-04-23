"""Package to produce rotating 3d plots."""
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Tuple, Union
from tqdm.auto import tqdm
import os
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool, cpu_count
from environments_utils import is_macos
import imageio
from pygifsicle import optimize
import shutil


def rotate_z(x: np.ndarray, y: np.ndarray, z: np.ndarray, theta: float):
    w = x+1j*y
    return np.vstack([
        np.real(np.exp(1j*theta)*w)/np.sqrt(2),
        np.imag(np.exp(1j*theta)*w)/np.sqrt(2),
        z/np.sqrt(2)
    ])


def _job(func: Callable, points: np.ndarray, theta: float, args, kwargs, path):
    fig, axes = func(
        rotate_z(points[0], points[1], points[2], theta), *args, **kwargs)
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


def rotate(
    func: Callable,
    points: np.ndarray,
    path: str,
    *args,
    fps: int = 24,
    duration: int = 1,
    cache_directory: str = ".rotate",
    parallelize: bool = True,
    verbose: bool = False,
    **kwargs
):
    """Create rotating gif of given image.

    Parameters
    -----------------------
    func: Callable,
        function return the figure.
    points: np.ndarray,
        The 3D array to rotate.
    path: str,
        path where to save the GIF.
    *args,
        positional arguments to be passed to the `func` callable.
    fps: int = 24,
        number of FPS to create.
    duration: int = 1,
        duration of the rotation in seconds.
    cache_directory: str = ".rotate",
        directory where to store the frame.
    parallelize: bool = True,
        whetever to parallelize execution.
    verbose: bool = False,
        whetever to be verbose about frame creation.
    **kwargs,
        keyword argument to be passed to the `func` callable

    """
    os.makedirs(cache_directory, exist_ok=True)
    X = MinMaxScaler(
        feature_range=(-1, 1)
    ).fit_transform(points.T).T

    tasks = [
        (
            func, X, 2*np.pi * frame / (duration*fps), args, kwargs,
            "{cache_directory}/{frame}.jpg".format(
                cache_directory=cache_directory,
                frame=frame
            )
        )
        for frame in range(duration*fps)
    ]

    if parallelize:
        with Pool(cpu_count()) as p:
            list(tqdm(
                p.imap(_job_wrapper, tasks),
                total=len(tasks),
                desc="Rendering frames",
                disable=not verbose
            ))
            p.close()
            p.join()
    else:
        for task in tqdm(tasks, desc="Rendering frames", disable=not verbose):
            _job_wrapper(task)

    with imageio.get_writer(path, mode='I', fps=fps) as writer:
        for task in tqdm(tasks, desc="Merging frames", disable=not verbose):
            writer.append_data(imageio.imread(task[-1]))

    optimize(path)
    shutil.rmtree(cache_directory)
