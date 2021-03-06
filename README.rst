ddd_subplots
=========================================================================================
|travis| |sonar_quality| |sonar_maintainability| |codacy| |code_climate_maintainability| |pip| |downloads|

Python package making it easier to handle mixed 3d and 2d subplots.

How do I install this package?
----------------------------------------------
As usual, just download it using pip:

.. code:: shell

    pip install ddd_subplots

Tests Coverage
----------------------------------------------
Since some software handling coverages sometime get slightly different results, here's three of them:

|coveralls| |sonar_coverage| |code_climate_coverage|

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


    def my_func(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, *args, **kwargs):
        fig, axes = subplots(1, 3, figsize=(9, 3))
        axs = axes.flatten()
        axs[0].scatter(xs, ys, zs, **kwargs)
        axs[1].scatter(ys, zs, xs, **kwargs)
        axs[2].scatter(zs, xs, ys, **kwargs)
        fig.tight_layout()
        return fig, axes


    X, y = datasets.load_iris(return_X_y=True)
    X_reduced = PCA(n_components=3).fit_transform(X)
    colors = np.array(["red", "green", "blue"])[y]
    rotate(my_func, *X_reduced.T, path="test.gif",
        duration=2, fps=24, c=colors, marker='o', s=20)



Output:

.. image:: https://github.com/LucaCappelletti94/ddd_subplots/blob/master/test.gif?raw=true


Known limits
----------------------------------------------
There is an error with `CoreFoundations and multiprocessing on MacOS <https://turtlemonvh.github.io/python-multiprocessing-and-corefoundation-libraries.html>`_, which states the following:

.. code:: bash

    The process has forked and you cannot use this CoreFoundation functionality safely. You MUST exec().
    Break on __THE_PROCESS_HAS_FORKED_AND_YOU_CANNOT_USE_THIS_COREFOUNDATION_FUNCTIONALITY___YOU_MUST_EXEC__() to debug.

This is a weird known error of MacOS Sierra. For now, the only available solution is to disable multiprocessing when dealing with matplotlib.
Any alternative valid solutions are welcome:

.. code:: python

    rotate(my_func, *X_reduced.T, path="test.gif",
        duration=2, fps=24, parallelize=False, c=colors, marker='o', s=20)

.. |travis| image:: https://travis-ci.org/LucaCappelletti94/ddd_subplots.png
   :target: https://travis-ci.org/LucaCappelletti94/ddd_subplots
   :alt: Travis CI build

.. |sonar_quality| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_ddd_subplots&metric=alert_status
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_ddd_subplots
    :alt: SonarCloud Quality

.. |sonar_maintainability| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_ddd_subplots&metric=sqale_rating
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_ddd_subplots
    :alt: SonarCloud Maintainability

.. |sonar_coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_ddd_subplots&metric=coverage
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_ddd_subplots
    :alt: SonarCloud Coverage

.. |coveralls| image:: https://coveralls.io/repos/github/LucaCappelletti94/ddd_subplots/badge.svg?branch=master
    :target: https://coveralls.io/github/LucaCappelletti94/ddd_subplots?branch=master
    :alt: Coveralls Coverage

.. |pip| image:: https://badge.fury.io/py/ddd-subplots.svg
    :target: https://badge.fury.io/py/ddd-subplots
    :alt: Pypi project

.. |downloads| image:: https://pepy.tech/badge/ddd-subplots
    :target: https://pepy.tech/badge/ddd-subplots
    :alt: Pypi total project downloads 

.. |codacy|  image:: https://api.codacy.com/project/badge/Grade/07125d5f5f4d4d1a838349b004553cd4
    :target: https://www.codacy.com/manual/LucaCappelletti94/ddd_subplots?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=LucaCappelletti94/ddd_subplots&amp;utm_campaign=Badge_Grade
    :alt: Codacy Maintainability

.. |code_climate_maintainability| image:: https://api.codeclimate.com/v1/badges/5c07f15635098d958e08/maintainability
    :target: https://codeclimate.com/github/LucaCappelletti94/ddd_subplots/maintainability
    :alt: Maintainability

.. |code_climate_coverage| image:: https://api.codeclimate.com/v1/badges/5c07f15635098d958e08/test_coverage
    :target: https://codeclimate.com/github/LucaCappelletti94/ddd_subplots/test_coverage
    :alt: Code Climate Coverate