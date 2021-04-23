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

conversion_command = """ffmpeg -framerate {fps}  -i "{path}/%d.jpg" -crf 20 -tune animation -preset veryslow -pix_fmt yuv444p10le {output_path} -y"""


def rotate_along_z_axis(x: np.ndarray, y: np.ndarray, z: np.ndarray, theta: float) -> Tuple[np.ndarray]:
    """Return points rotate along z-axis.

    Parameters
    ---------------------
    x: np.ndarray,
        First axis of the points vector.
    y: np.ndarray,
        Second axis of the points vector.
    z: np.ndarray,
        Third axis of the points vector.
        This is the axis that will be the rotation axis.
    theta: float,
        Theta for the current variation.

    Returns
    ----------------------
    Tuple with rotated values.
    """
    w = x+1j*y
    return (
        np.real(np.exp(1j*theta)*w)/np.sqrt(2),
        np.imag(np.exp(1j*theta)*w)/np.sqrt(2),
        z/np.sqrt(2)
    )


def rotating_spiral(x: np.ndarray, y: np.ndarray, z: np.ndarray, theta: float) -> np.ndarray:
    """Return rotated points following a spiral path.

    Parameters
    ---------------------
    x: np.ndarray,
        First axis of the points vector.
    y: np.ndarray,
        Second axis of the points vector.
    z: np.ndarray,
        Third axis of the points vector.
        This is the axis that will be the rotation axis.
    theta: float,
        Theta for the current variation.

    Returns
    ----------------------
    Numpy array with rotated values.
    """
    x, y, z = rotate_along_z_axis(x, y, z, theta)
    x, z, y = rotate_along_z_axis(x, z, y, theta*2)
    z, y, x = rotate_along_z_axis(z, y, x, theta*4)
    return np.vstack([x, y, z])


def _job(func: Callable, points: np.ndarray, theta: float, args, kwargs, path):
    fig, _ = func(
        rotating_spiral(
            points[0],
            points[1],
            points[2],
            theta
        ),
        *args,
        **kwargs
    )
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
    global conversion_command

    is_gif = path.endswith(".gif")

    if not is_gif and shutil.which("ffmpeg") is None:
        raise ValueError((
            "The path required is not a gif, so it will be built as a video "
            "using ffmpeg, but it was not found installed in the system."
        ))

    os.makedirs(cache_directory, exist_ok=True)
    X = MinMaxScaler(
        feature_range=(-1, 1)
    ).fit_transform(points.T).T

    total_frames = duration*fps

    tasks = [
        (
            func, X, 2 * np.pi * frame / total_frames, args, kwargs,
            "{cache_directory}/{frame}.jpg".format(
                cache_directory=cache_directory,
                frame=frame
            )
        )
        for frame in range(total_frames)
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

    if is_gif:
        with imageio.get_writer(path, mode='I', fps=fps) as writer:
            for task in tqdm(tasks, desc="Merging frames", disable=not verbose):
                writer.append_data(imageio.imread(task[-1]))

        optimize(path)
    else:
        os.system(conversion_command.format(
            fps=fps,
            output_path=path,
            path=cache_directory
        ))
    shutil.rmtree(cache_directory)
