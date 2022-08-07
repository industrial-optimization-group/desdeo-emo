[![PyPI version](https://badge.fury.io/py/desdeo-tools.svg)](https://badge.fury.io/py/desdeo-tools)
[![Documentation Status](https://readthedocs.org/projects/desdeo-tools/badge/?version=latest)](https://desdeo-tools.readthedocs.io/en/latest/?badge=latest)

# desdeo-tools

Generic tools and design language used in the
[DESDEO framework](https://github.com/industrial-optimization-group/DESDEO).
For example,
this package contains classes for interacting with optimization methods
implemented in `desdeo-mcmd` and `desdeo-emo`, and tools for solving for a
representation of a Pareto optimal front for a multiobjective optimization.

## Installation

### For regular users
You can install the `desdeo-tools` package from the Python package index using `pip` by issuing the command `pip install desdeo-tools`.

### For developers
Requires [poetry](https://python-poetry.org/). See `pyproject.toml` for Python package requirements.

To install and use the this package with poetry on a \*nix based system,
issue the following commands:

1. `git clone https://github.com/industrial-optimization-group/desdeo-tools`
2. `cd desdeo-tools`
3. `poetry init`
4. `poetry install`

## Documentation

Documentation and usage examples for this package can be found [here](https://desdeo-tools.readthedocs.io/en/latest/)

## Citation

If you decide to use DESDEO is any of your works or research, we would appreciate you citing the appropiate paper published in [IEEE Access](https://doi.org/10.1109/ACCESS.2021.3123825) (open access).
