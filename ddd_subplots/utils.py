from typing import Optional
import os
from environments_utils import is_notebook


def display_video_at_path(path: str, width: str = "480", height: Optional[str] = "480"):
    """Returns IPython object to properly display video at given path.

    Parameters
    -------------------------
    path: str
    width: str = "480"
    height: Optional[str] = "480"
    """
    if not os.path.exists(path):
        raise ValueError(
            "The file at given path `{}` does not exist.".format(path)
        )
    if is_notebook():
        from IPython.display import HTML
        return HTML(
            (
                "<center>"
                "<video width=\"{width}\"{height} src=\"{path}\" autoplay muted playsinline loop>"
                "<source src=\"{path}\" type=\"video/{extension}\" />"
                "Your browser does not support the video tag."
                "</video>"
                "</center>"
            ).format(
                path=path,
                width=width,
                height="" if height is None else " height=\"{height}\"".format(
                    height=height
                ),
                extension=path.split(".")[-1]
            )
        )
