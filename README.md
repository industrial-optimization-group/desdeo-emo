# desdeo-emo
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/industrial-optimization-group/desdeo-emo/master)
[![Tests](https://github.com/Sepuliini/desdeo-emo/actions/workflows/automate-DESDEO-emo.yml/badge.svg)](https://github.com/Sepuliini/desdeo-emo/actions/workflows/automate-DESDEO-emo.yml)

[![PyPI version](https://badge.fury.io/py/desdeo-emo.svg)](https://badge.fury.io/py/desdeo-emo)
[![Documentation Status](https://readthedocs.org/projects/desdeo-emo/badge/?version=latest)](https://desdeo-emo.readthedocs.io/en/latest/?badge=latest)


The evolutionary algorithms package within the [DESDEO framework](https://github.com/industrial-optimization-group/DESDEO).

Code for the SoftwareX paper can be found in [this notebook](docs/notebooks/Using_EvoNN_for_optimization.ipynb).

Currently supported:
* Multi-objective optimization with visualization and interaction support.
* Preference is accepted as a reference point.
* Surrogate modelling (neural networks and genetic trees) evolved via EAs.
* Surrogate assisted optimization
* Constraint handling using `RVEA`
* IOPIS optimization using `RVEA` and `NSGA-III`

Currently _NOT_ supported:
* Binary and integer variables.

To test the code, open the [binder link](https://mybinder.org/v2/gh/industrial-optimization-group/desdeo-emo/master) and read example.ipynb.

Read the documentation [here](https://desdeo-emo.readthedocs.io/en/latest/)

### Requirements
* Python 3.9 or 3.10.
* [Poetry dependency manager](https://github.com/sdispater/poetry): Only for developers

### Installation process for normal users
* Create a new virtual enviroment for the project
* Run: `pip install desdeo_emo`

### Installation process for developers
* Download and extract the code or `git clone`
* Create a new virtual environment for the project
* Run `poetry install` inside the virtual environment shell.

## Citation

If you decide to use DESDEO is any of your works or research, we would appreciate you citing the appropiate paper published in [IEEE Access](https://doi.org/10.1109/ACCESS.2021.3123825) (open access).
