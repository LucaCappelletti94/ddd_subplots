"""Package to produce rotating 3d plots."""
import os
import warnings
from typing import Callable, List, Union

import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
from matplotlib.axes import Axes
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import trange


def rotate_along_last_axis(
    x: np.ndarray, y: np.ndarray, *features: List[np.ndarray], theta: float
) -> List[np.ndarray]:
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
    w = x + 1j * y
    return [
        np.real(np.exp(1j * theta) * w),
        np.imag(np.exp(1j * theta) * w),
        *[feature for feature in features],
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
        new_features = rotate_along_last_axis( # pylint: disable=no-value-for-parameter
            *features, theta=theta * min(2**i, 2)
        )
        features[-1] = new_features[0]
        features[:-1] = new_features[1:]
    return np.vstack([feature / np.sqrt(2) for feature in features])


def render_frame(
    func: Callable,
    points: Union[np.ndarray, List[np.ndarray]],
    theta: float,
    args: List,
    **kwargs,
) -> np.ndarray:
    """Returns rendered frame.

    Parameters
    -----------------------
    func: Callable,
        Function to call to renderize the frame.
    points: Union[np.ndarray, List[np.ndarray]],
        The points to be rotated and renderized.
    theta: float,
        The amount of rotation.
    args: List,
        The list of positional arguments.
    kwargs: Dict,
        The dictionary of keywargs arguments.
    """
    points = [rotating_spiral(*matrix.T, theta=theta).T for matrix in points]

    points = [matrix[:, :3] if matrix.shape[1] > 3 else matrix for matrix in points]

    returned_value = func(points[0] if len(points) == 1 else points, *args, **kwargs)

    if not isinstance(returned_value, tuple):
        raise ValueError(
            "The provided rendering function does not return "
            "a tuple with figure and axes!"
        )

    fig, axis = returned_value[:2]
    canvas = FigureCanvas(fig)

    window = 1.0
    if any([matrix.shape[1] > 2 for matrix in points]):
        window = 0.6

    if isinstance(axis, (Axes, Axis)):
        axis = np.array([axis])

    for ax in axis.flatten():
        ax.set_axis_off()
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xlim(-window, window)
        ax.set_ylim(-window, window)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.axis("off")
        try:
            ax.set_zlim(-window, window)
            ax.set_zticklabels([])
        except AttributeError:
            pass

    canvas.draw()  # draw the canvas, cache the renderer

    width, height = fig.get_size_inches() * fig.get_dpi()

    data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(
        int(height), int(width), 3
    )

    plt.close(fig)
    plt.close()

    return data


def rotate(
    func: Callable,
    points: Union[np.ndarray, List[np.ndarray]],
    path: str,
    *args,
    fps: int = 24,
    duration: int = 1,
    verbose: bool = False,
    **kwargs,
):
    """Create rotating gif of given image.

    Parameters
    -----------------------
    func: Callable
        function return the figure.
    points: Union[np.ndarray, List[np.ndarray]]
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

    Raises
    -----------------------
    ValueError
        If the provided points cloud is None.
    ValueError
        If the provided points cloud is not a numpy array.
    """
    if os.path.dirname(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        os.remove(path)

    if not isinstance(points, list):
        points = [points]

    for i, points_cloud in enumerate(points):
        if points_cloud is None:
            raise ValueError(f"The provided points cloud at index {i} is None!")
        if not isinstance(points_cloud, np.ndarray):
            raise ValueError(
                f"The provided points cloud at index {i} is not a numpy array! "
                f"Instead it is a {type(points_cloud)}!"
            )

    scaled_points = [
        MinMaxScaler(feature_range=(-1, 1)).fit_transform(matrix) for matrix in points
    ]

    total_frames = duration * fps

    is_gif = path.endswith(".gif")
    is_video = path.split(".")[-1] in ("webm", "mp4", "avi")

    if is_gif:
        gif_writer = imageio.get_writer(path, mode="I", fps=fps)
    elif is_video:
        encoding = {"mp4": "MP4V", "avi": "FMP4", "webm": "vp80"}[path.split(".")[-1]]
        fourcc = cv2.VideoWriter_fourcc(*encoding)  # pylint: disable=no-member
    else:
        raise ValueError(
            "The provided format, as detected from the provided "
            "path extension, is not supported! "
            f"The path you have provided is `{path}`."
        )

    for frame in trange(
        total_frames,
        desc="Rendering",
        disable=not verbose,
        dynamic_ncols=True,
        leave=False,
    ):
        rendered_frame = render_frame(
            func=func,
            points=scaled_points,
            theta=2 * np.pi * frame / total_frames,
            args=args,
            **kwargs,
        )

        if is_gif:
            gif_writer.append_data(rendered_frame)
        else:
            # If this is the first frame
            if frame == 0:
                height, width, _ = rendered_frame.shape
                video_writer = cv2.VideoWriter(  # pylint: disable=no-member
                    path, fourcc, fps, (width, height)
                )
            video_writer.write(rendered_frame)

    if is_gif:
        try:
            from pygifsicle import optimize  # pylint: disable=import-outside-toplevel

            optimize(path)
        except ImportError:
            warnings.warn(
                "The `pygifsicle` package is not installed. "
                "It is not possible to optimize the GIF "
                "file size, which might be very large. "
                "Considering installing it with `pip install pygifsicle`, "
                "which will require `gifsicle` to be installed "
                "in your system."
            )
    else:
        cv2.destroyAllWindows()  # pylint: disable=no-member
        video_writer.release()

    if not os.path.exists(path):
        raise ValueError(
            (
                f"The expected target path file `{path}` was "
                "not created. Tipically this is caused by some "
                "errors in the encoding of the file that has "
                "been chosen. Please take a look at the log that "
                "has be printed in either the console or the jupyter "
                "kernel."
            )
        )
