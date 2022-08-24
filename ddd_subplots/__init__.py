"""Module providing utilities to animate rotations of multidimensional objects."""
from support_developer import support_luca
from .subplots import subplots
from .rotate import rotate
from .utils import display_video_at_path

support_luca("ddd_subplots")

__all__ = ["subplots", "rotate", "display_video_at_path"]
