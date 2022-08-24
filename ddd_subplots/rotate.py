"""Package to produce rotating 3d plots."""
import os
from typing import Callable, Dict, List, Optional

import imageio
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.axes import Axes
import numpy as np
import cv2
from pygifsicle import optimize
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import trange


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
            theta=theta*min(2**i, 2)
        )
        features[-1] = new_features[0]
        features[:-1] = new_features[1:]
    return np.vstack([
        feature / np.sqrt(2)
        for feature in features
    ])


def render_frame(
    func: Callable,
    points: np.ndarray,
    theta: float,
    args: List,
    path: str,
    **kwargs,
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
    path: str,
        The path where to save the frame.
    kwargs: Dict,
        The dictionary of keywargs arguments.
    """
    points = rotating_spiral(
        *points.T,
        theta=theta
    ).T

    if points.shape[1] > 3:
        points = points[:, :3]

    returned_value = func(
        points,
        *args,
        **kwargs
    )

    if not isinstance(returned_value, tuple):
        raise ValueError(
            "The provided rendering function does not return "
            "a tuple with figure and axes!"
        )

    fig, axis = returned_value

    window = 1.0
    if points.shape[1] > 2:
        window = 0.6

    if isinstance(axis, (Axes, Axis)):
        axis = np.array([axis])

    for ax in axis.flatten():
        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(-window, window)
        ax.set_ylim(-window, window)
        try:
            ax.set_zlim(-window, window)
            ax.set_zticklabels([])
        except AttributeError:
            pass

    fig.savefig(path)
    plt.close(fig)


def rotate(
    func: Callable,
    points: np.ndarray,
    path: str,
    *args,
    fps: int = 24,
    duration: int = 1,
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
    verbose: bool = False
        whetever to be verbose about frame creation.
    **kwargs
        keyword argument to be passed to the `func` callable

    """
    global conversion_command

    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        os.remove(path)

    scaled_points = MinMaxScaler(
        feature_range=(-1, 1)
    ).fit_transform(points)

    total_frames = duration*fps

    is_gif = path.endswith(".gif")
    is_video = path.split(".")[-1] in ("webm", "mp4", "avi")

    if is_gif:
        gif_writer = imageio.get_writer(path, mode='I', fps=fps)
    elif is_video:
        encoding = {
            "mp4": "MP4V",
            "avi": "FMP4",
            "webm": "vp80"
        }[path.split(".")[-1]]
        fourcc = cv2.VideoWriter_fourcc(*encoding)
    else:
        raise ValueError("Unsupported format!")

    for frame in trange(
        total_frames,
        desc="Rendering",
        disable=not verbose,
        dynamic_ncols=True,
        leave=False
    ):
        frame_path = "{path}.{frame}.tmp.jpg".format(
            path=path,
            frame=frame
        )

        rate = frame / total_frames

        render_frame(
            func=func,
            points=scaled_points,
            theta=2 * np.pi * rate,
            args=args,
            path=frame_path,
            **kwargs
        )

        if is_gif:
            gif_writer.append_data(imageio.imread(frame_path))
        else:
            # If this is the first frame
            if frame == 0:
                height, width, _ = cv2.imread(frame_path).shape
                video_writer = cv2.VideoWriter(
                    path, fourcc, fps, (width, height))
            video_writer.write(cv2.imread(frame_path))

        # And we clean up the path.
        os.remove(frame_path)

    if is_gif:
        optimize(path)
    else:
        cv2.destroyAllWindows()
        video_writer.release()

    if not os.path.exists(path):
        raise ValueError(
            (
                "The expected target path file `{}` was "
                "not created. Tipically this is caused by some "
                "errors in the encoding of the file that has "
                "been chosen. Please take a look at the log that "
                "has be printed in either the console or the jupyter "
                "kernel."
            ).format(path)
        )
