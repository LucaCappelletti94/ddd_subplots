"""Package to produce rotating 3d plots."""
import os
import shutil
from multiprocessing import Pool, cpu_count
from typing import Callable, Dict, List, Tuple, Any

import imageio
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.arraysetops import isin
from pygifsicle import optimize
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm

conversion_command = """ffmpeg -framerate {fps}  -i "{path}/%d.jpg" -crf 20 -tune animation -preset veryslow -pix_fmt yuv444p10le {output_path} -y"""


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def rotate_along_last_axis(x: np.ndarray, y: np.ndarray, *features: List[np.ndarray], theta: float) -> List[np.ndarray]:
    """Return points rotate along z-axis.

    Parameters
    ---------------------
    x: np.ndarray,
        First axis of the points vector.
    y: np.ndarray,
        Second axis of the points vector.
    features: List[np.ndarray],
        Extra features to be rotated.
    theta: float,
        Theta for the current variation.

    Returns
    ----------------------
    Tuple with rotated values.
    """
    w = x+1j*y
    return [
        np.real(np.exp(1j*theta)*w),
        np.imag(np.exp(1j*theta)*w),
        *[
            feature
            for feature in features
        ]
    ]


def rotating_spiral(*features: List[np.ndarray], theta: float) -> np.ndarray:
    """Return rotated points following a spiral path.

    Parameters
    ---------------------
    features: List[np.ndarray],
        Extra features to be rotated.
    theta: float,
        Theta for the current variation.

    Returns
    ----------------------
    Numpy array with rotated values.
    """
    features = list(features)
    for i in range(len(features)):
        new_features = rotate_along_last_axis(
            *features,
            theta=theta*min(2**i, 4)
        )
        features[-1] = new_features[0]
        features[:-1] = new_features[1:]
    return np.vstack([
        feature / np.sqrt(2)
        for feature in features
    ])


def _render_frame(
    func: Callable,
    points: np.ndarray,
    theta: float,
    args: List,
    kwargs: Dict,
    path: str
):
    """Method for rendering frame.

    Parameters
    -----------------------
    func: Callable,
        Function to call to renderize the frame.
    points: np.ndarray,
        The points to be rotated and renderized.
    theta: float,
        The amount of rotation.
    args: List,
        The list of positional arguments.
    kwargs: Dict,
        The dictionary of keywargs arguments.
    path: str,
        The path where to save the frame.
    """
    points = rotating_spiral(
        *points.T,
        theta=theta
    ).T

    if points.shape[1] > 3:
        points = points[:, :3]

    fig, axis = func(
        points,
        *args,
        **kwargs
    )
    axis.set_axis_off()
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_xlim(-1, 1)
    axis.set_ylim(-1, 1)
    try:
        axis.set_zlim(-1, 1)
        axis.set_zticklabels([])
    except AttributeError:
        pass
    fig.savefig(path)
    plt.close(fig)


def _render_frame_wrapper(tasks: List[Tuple]) -> int:
    """Wrapper method for rendering frame."""
    for task in tasks:
        _render_frame(*task)
    return len(tasks)


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
    func: Callable
        function return the figure.
    points: np.ndarray
        The 3D or 4D array to rotate or roto-translate.
    path: str
        path where to save the GIF.
    *args
        positional arguments to be passed to the `func` callable.
    fps: int = 24
        number of FPS to create.
    duration: int = 1
        Duration of the rotation in seconds.
    cache_directory: str = ".rotate"
        directory where to store the frame.
    parallelize: bool = True
        whetever to parallelize execution.
    verbose: bool = False
        whetever to be verbose about frame creation.
    **kwargs
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
    ).fit_transform(points)

    total_frames = duration*fps

    tasks = [
        (
            func,
            X,
            2 * np.pi * frame / total_frames,
            args,
            kwargs,
            "{cache_directory}/{frame}.jpg".format(
                cache_directory=cache_directory,
                frame=frame
            )
        )
        for frame in range(total_frames)
    ]

    if parallelize:
        number_of_processes = cpu_count()
        with Pool(number_of_processes) as p:
            chunks_size = total_frames // number_of_processes
            loading_bar = tqdm(
                total=total_frames,
                desc="Rendering frames",
                disable=not verbose,
                dynamic_ncols=True,
                leave=False
            )
            for executed_tasks_number in p.imap(_render_frame_wrapper, chunks(tasks, chunks_size)):
                loading_bar.update(executed_tasks_number)
            loading_bar.close()
            p.close()
            p.join()
    else:
        for task in tqdm(tasks, desc="Rendering frames", disable=not verbose, dynamic_ncols=True, leave=False):
            _render_frame_wrapper([task])

    if is_gif:
        with imageio.get_writer(path, mode='I', fps=fps) as writer:
            for task in tqdm(tasks, desc="Merging frames", disable=not verbose, dynamic_ncols=True, leave=False):
                writer.append_data(imageio.imread(task[-1]))
        optimize(path)
    else:
        os.system(conversion_command.format(
            fps=fps,
            output_path=path,
            path=cache_directory
        ))
    shutil.rmtree(cache_directory)
