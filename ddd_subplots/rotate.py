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


def sliding_space(
    points: np.ndarray,
    args: List[Any],
    kwargs: Dict[str, Any],
    threshold: float,
    auto_slice: bool,
    epsilon: float = 0.05
) -> np.ndarray:
    """Return given points sliced in the fourth dimension into the bucket of given size.

    Parameters
    ------------------------
    points: np.ndarray
        The points to be sliced.
    args: List,
        The list of positional arguments.
    kwargs: Dict,
        The dictionary of keywargs arguments.
    threshold: float
        The current threshold position to be used.
    auto_slice: bool = True,
        Whether to automatically slice all other numpy vector
        objects provided alongside the points if they have the same
        number of samples.
    epsilon: float = 0.05
        Size of the window to take into consideration
    """
    assert points.shape[1] == 4
    mask = (
        points[:, 3] >= threshold - epsilon
    ) & (
        points[:, 3] <= threshold + epsilon
    )
    # If this is the threshold on the lower side
    # we want to include the points on the other side
    # so to create a looper-around dimension.
    if threshold - epsilon < -1:
        mask |= points[:, 3] >= 2 - (threshold - epsilon)
    # Analogously if we are on the higher side
    if threshold + epsilon > 1:
        mask |= points[:, 3] <= (threshold + epsilon) - 2
    if auto_slice:
        args = [
            arg[mask]
            if isinstance(arg, np.ndarray) and arg.shape[0] == points.shape[0]
            else arg
            for arg in args
        ]
        kwargs = {
            kw: arg[mask]
            if isinstance(arg, np.ndarray) and arg.shape[0] == points.shape[0]
            else arg
            for kw, arg in kwargs.items()
        }

    # The we apply the mask and slice the points
    return points[mask][:, :3], args, kwargs


def _render_frame(
    func: Callable,
    points: np.ndarray,
    threshold: float,
    theta: float,
    auto_slice: bool,
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
    threshold: float
        The current threshold position to be used.
    theta: float,
        The amount of rotation.
    auto_slice: bool = True
        Whether to automatically slice all other numpy vector
        objects provided alongside the points if they have the same
        number of samples.
    args: List,
        The list of positional arguments.
    kwargs: Dict,
        The dictionary of keywargs arguments.
    path: str,
        The path where to save the frame.
    """
    if points.shape[1] == 4:
        points, args, kwargs = sliding_space(
            points,
            args,
            kwargs,
            threshold=threshold,
            auto_slice=auto_slice
        )
    fig, axis = func(
        rotating_spiral(
            *points.T,
            theta
        ).T,
        *args,
        **kwargs
    )
    axis.set_axis_off()
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_zticklabels([])
    axis.set_xlim(-0.25, 0.25)
    axis.set_ylim(-0.25, 0.25)
    axis.set_zlim(-0.25, 0.25)
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
    auto_slice: bool = True,
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
    auto_slice: bool = True,
        Whether to automatically slice all other numpy vector
        objects provided alongside the points if they have the same
        number of samples.
    verbose: bool = False
        whetever to be verbose about frame creation.
    **kwargs
        keyword argument to be passed to the `func` callable

    """
    if points.shape[1] not in (3, 4):
        raise ValueError(
            (
                "In order to draw a 3D rotating plot, you need to provide "
                "a 3D or 4D matrix. The one you have provided has shape `{}`."
            ).format(points.shape)
        )
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
            2*frame / total_frames - 1,
            2 * np.pi * frame / total_frames,
            auto_slice,
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
