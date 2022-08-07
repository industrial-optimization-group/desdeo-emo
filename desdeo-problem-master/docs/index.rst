.. desdeo-problem documentation master file, created by
   sphinx-quickstart on Wed Jun 17 11:04:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to desdeo-problem's documentation
=========================================

Contains tools to model and define multiobjective optimization problems to be used in the DESDEO framework.

Requirements
============

* Python 3.7 or newer.
* `Poetry dependency manager <https://python-poetry.org/>`__ : Only for developers.

See `pyproject.toml` for Python package requirements.


Installation
============

To install and use this package on a \*nix-based system, follow one of the following procedures.


For users
---------


First, create a new virtual environment for the project. Then install the package using the following command:

::

    $ pip install desdeo_problem




For developers
--------------


Download the code or clone it with the following command:

::

    $ git clone https://github.com/industrial-optimization-group/desdeo-problem

Then, create a new virtual environment for the project and install the package in it:

::

    $ cd desdeo-problem
    $ poetry init
    $ poetry install

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents

   api
   examples


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
