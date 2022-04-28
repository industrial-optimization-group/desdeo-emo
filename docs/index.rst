.. desdeo-emo documentation master file, created by
   sphinx-quickstart on Sun Aug 23 18:32:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to desdeo-emo's documentation!
======================================

The evolutionary algorithms package within the `desdeo` framework.

Currently supported:

* Multi-objective optimization with visualization and interaction support.
* Preference is accepted as a reference point.
* Surrogate modelling (neural networks and genetic trees) evolved via EAs.
* Surrogate assisted optimization
* Constraint handling using `RVEA`
* IOPIS optimization using `RVEA` and `NSGA-III`

Currently **NOT** supported:

* Binary and integer variables.

To test the code, open the `binder link <https://mybinder.org/v2/gh/industrial-optimization-group/desdeo-emo/master>`__ and read example.ipynb.


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

    $ pip install desdeo-emo




For developers
--------------


Download the code or clone it with the following command:

::

    $ git clone https://github.com/industrial-optimization-group/desdeo-emo

Then, create a new virtual environment for the project and install the package in it:

::

    $ cd desdeo-emo
    $ poetry init
    $ poetry install


Currently implemented methods
=============================


.. table:: 
   :widths: 20 80

   ====================  ========================================================================================================================================
     Algorithm           Reference
   ====================  ========================================================================================================================================
   **RVEA**              R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff, 
		         A Reference Vector Guided Evolutionary Algorithm for Many-objective Optimization, 
		         IEEE Transactions on Evolutionary Computation, 2016
   **NSGA-III**          K. Deb and H. Jain, "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach, 
		         Part I: Solving Problems With Box Constraints," in IEEE Transactions on Evolutionary Computation, 
			 vol. 18, no. 4, pp. 577-601, Aug. 2014, doi: 10.1109/TEVC.2013.2281535.
   **MOEA/D**            Q. Zhang and H. Li, "MOEA/D: A Multiobjective Evolutionary Algorithm Based on Decomposition," 
                         in IEEE Transactions on Evolutionary Computation, vol. 11, no. 6, pp. 712-731, Dec. 2007, doi: 10.1109/TEVC.2007.892759.
   **PPGA**              Laumanns, M., Rudolph, G., & Schwefel, H. P. (1998). A spatial predator-prey approach to multi-objective optimization: 
                         A preliminary study. In International Conference on Parallel Problem Solving from Nature (pp. 241-249). Springer, Berlin, Heidelberg.
   **IOPIS-RVEA**        Saini B.S., Hakanen J., Miettinen K. (2020) A New Paradigm in Interactive Evolutionary Multiobjective Optimization. 
                         In: Bäck T. et al. (eds) Parallel Problem Solving from Nature – PPSN XVI. PPSN 2020. 
                         Lecture Notes in Computer Science, vol 12270. Springer, Cham. https://doi.org/10.1007/978-3-030-58115-2_17
   **IOPIS-NSGA-III**    Saini B.S., Hakanen J., Miettinen K. (2020) A New Paradigm in Interactive Evolutionary Multiobjective Optimization. 
                         In: Bäck T. et al. (eds) Parallel Problem Solving from Nature – PPSN XVI. PPSN 2020.
                         Lecture Notes in Computer Science, vol 12270. Springer, Cham. https://doi.org/10.1007/978-3-030-58115-2_17
   ====================  ========================================================================================================================================

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents

   README <README>
   background
   algorithms
   operators
   api
   examples


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
