"""Submodule providing 3d subplots handler."""
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


def subplots(
    nrows: int = 1,
    ncols: int = 1,
    subplot_kw: Dict = None,
    squeeze: bool = True,
    ** fig_kw: Dict
) -> Tuple[Figure, Axes3D]:
    """
    Create a figure and a set of subplots.

    This utility wrapper makes it convenient to create common layouts of
    subplots, including the enclosing figure object, in a single call.

    Parameters
    ----------
    nrows, ncols : int, optional, default: 1
        Number of rows/columns of the subplot grid.
    subplot_kw : dict, optional
        Dict with keywords passed to the
        `~matplotlib.figure.Figure.add_subplot` call used to create each
        subplot.
    **fig_kw
        All additional keyword arguments are passed to the
        `.pyplot.figure` call.

    Returns
    -------
    fig : `~.figure.Figure`

    ax : `.axes.Axes` object or array of Axes objects.
        *ax* can be either a single `~matplotlib.axes.Axes` object or an
        array of Axes objects if more than one subplot was created.  The
        dimensions of the resulting array can be controlled with the squeeze
        keyword, see above.
    """
    fig = plt.figure(**fig_kw)
    axes = np.array([
        fig.add_subplot(
            nrows,
            ncols,
            1+i,
            **({} if subplot_kw is None else subplot_kw),
            projection='3d'
        )
        for i in range(nrows*ncols)
    ]).reshape(nrows, ncols)
    if squeeze and ncols == 1 and nrows == 1:
        axes = axes.flatten()[0]
    return fig, axes
