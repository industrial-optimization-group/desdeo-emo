# desdeo-emo
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/industrial-optimization-group/desdeo-emo/master)

The evolutionary algorithms package within the `desdeo` framework.

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
* Python 3.7 (3.8 is NOT supported at the moment)
* [Poetry dependency manager](https://github.com/sdispater/poetry): Only for developers

### Installation process for normal users
* Create a new virtual enviroment for the project
* Run: `pip install desdeo_emo`

### Installation process for developers
* Download and extract the code or `git clone`
* Create a new virtual environment for the project
* Run `poetry install` inside the virtual environment shell.