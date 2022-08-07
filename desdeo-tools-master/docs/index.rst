.. desdeo-tools documentation master file, created by
   sphinx-quickstart on Wed Jun 17 00:20:32 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to desdeo-tools' documentation! 
========================================

This package contains generic tools and design language used in the DESDEO framework. For example,
it includes classes for interacting with optimization methods
implemented in `desdeo-mcdm` and `desdeo-emo`, and tools for solving a
representation of a Pareto optimal front for a multiobjective optimization problem.

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

    $ pip install desdeo_tools




For developers
--------------


Download the code or clone it with the following command:

::

    $ git clone https://github.com/industrial-optimization-group/desdeo-tools

Then, create a new virtual environment for the project and install the package in it:

::

    $ cd desdeo-tools
    $ poetry init
    $ poetry install

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents

   examples


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
