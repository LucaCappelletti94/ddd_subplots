DDD Subplots
=========================================================================================
|pip| |downloads|

Python package making it easier to handle mixed 3d and 2d subplots.

How do I install this package?
----------------------------------------------
As usual, just download it using pip:

.. code:: shell

    pip install ddd_subplots

Usage Example
-----------------------------------------------

3D subplots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To get a set of 3d subplots just import `subplots`:

.. code:: python

    from ddd_subplots import subplots

    fig, axes = subplots(1, 3, figsize=(15, 5))


Rotating 3D scatter plots
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The library also offers a method to render 3D scatter plots. Here's a complete example:

.. code:: python

    from ddd_subplots import subplots, rotate
    import numpy as np
    from sklearn import datasets
    from sklearn.decomposition import PCA


    def write_frame(X_reduced, y):
        colors = np.array(["red", "green", "blue"])[y]
        fig, axes = subplots(1, 3, figsize=(15, 5))
        for axis in axes.flatten():
            axis.scatter(*X_reduced.T, depthshade=False,
                        c=colors, marker='o', s=20)
        fig.tight_layout()
        return fig, axes

    X, y = datasets.load_iris(return_X_y=True)
    X_reduced = PCA(n_components=3).fit_transform(X)

    rotate(
        write_frame,
        X_reduced,
        "test_animation.gif",
        y,
        duration=10,
        verbose=True
    )


Output:

.. image:: https://github.com/LucaCappelletti94/ddd_subplots/blob/master/test_animation.gif?raw=true


.. |pip| image:: https://badge.fury.io/py/ddd-subplots.svg
    :target: https://badge.fury.io/py/ddd-subplots
    :alt: Pypi project

.. |downloads| image:: https://pepy.tech/badge/ddd-subplots
    :target: https://pepy.tech/badge/ddd-subplots
    :alt: Pypi total project downloads 
