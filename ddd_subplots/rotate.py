from matplotlib.axis import Axis
import numpy as np
from typing import Callable, Tuple
from tqdm.auto import tqdm
import os
from multiprocessing import Pool, cpu_count
import imageio
from pygifsicle import optimize
import shutil


def rotate_z(x: np.ndarray, y: np.ndarray, z: np.ndarray, theta: float):
    w = x+1j*y
    return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), z


def _job(func: Callable, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, theta: float, args, kwargs, path):
    fig = func(*rotate_z(xs, ys, zs, theta), *args, **kwargs)
    fig.savefig(path)


def _job_wrapper(task: Tuple):
    _job(*task)


def rotate(func: Callable, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, path: str, fps: int = 24, duration: int = 1, cache_directory: str = ".rotate", verbose: bool = False, *args, **kwargs):
    os.makedirs(cache_directory, exist_ok=True)

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

    with Pool(cpu_count()) as p:
        list(tqdm(
            p.imap(_job_wrapper, tasks),
            total=len(tasks),
            desc="Rendering frames",
            disable=not verbose))

    with imageio.get_writer(path, mode='I', fps=fps) as writer:
        for task in tqdm(tasks, disable=not verbose):
            writer.append_data(imageio.imread(task[-1]))

    optimize(path)
    shutil.rmtree(cache_directory)
