"""Package to produce rotating 3d plots."""
import os
import shutil
from multiprocessing import Pool, cpu_count
from typing import Callable, Dict, List, Optional, Tuple

import imageio
import matplotlib.pyplot as plt
import numpy as np
from pygifsicle import optimize
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import tqdm, trange

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


def _render_frame(
    func: Callable,
    points: np.ndarray,
    theta: float,
    args: List,
    kwargs: Dict,
    path: str
) -> Optional[np.ndarray]:
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
    fig.tight_layout()

    if path is None:
        # Now we can save it to a numpy array.
        resulting_image = np.fromstring(
            fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        resulting_image = resulting_image.reshape(
            fig.canvas.get_width_height()[::-1] + (3,))
    else:
        # Or else we save it to file
        fig.savefig(path)
        resulting_image = None

    plt.close(fig)
    return resulting_image


def _render_frame_wrapper(task: Tuple) -> Optional[np.ndarray]:
    """Wrapper method for rendering frame."""
    return _render_frame(*task)


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
    if points.shape[1] != 3:
        raise ValueError(
            (
                "In order to draw a 3D rotating plot, you need to provide "
                "a 3D matrix. The one you have provided has shape `{}`."
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

    # Create the iterator over the tasks.
    tasks_iterator = (
        (
            func, X, 2 * np.pi * frame / total_frames, args, kwargs,
            (
                None if is_gif else
                "{cache_directory}/{frame}.jpg".format(
                    cache_directory=cache_directory,
                    frame=frame
                )
            )
        )
        for frame in trange(
            total_frames,
            desc="Rendering frames",
            disable=not verbose,
            leave=False,
            dynamic_ncols=True
        )
    )
    if parallelize:
        pool = Pool(cpu_count())
        tasks_iterator = pool.imap(_render_frame_wrapper, tasks_iterator)
    else:
        tasks_iterator = (
            _render_frame_wrapper(task)
            for task in tasks_iterator
        )

    if is_gif:
        # Otherwise we return the images.
        with imageio.get_writer(path, mode='I', fps=fps) as writer:
            for image in tasks_iterator:
                writer.append_data(image)
        optimize(path)
    else:
        # We blindly consume the iterator.
        for _ in tasks_iterator:
            pass
        os.system(conversion_command.format(
            fps=fps,
            output_path=path,
            path=cache_directory
        ))
        shutil.rmtree(cache_directory)

    # If we have started the pool we need to close it down.
    if parallelize:
        pool.close()
        pool.join()
