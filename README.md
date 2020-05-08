# desdeo-emo
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/industrial-optimization-group/desdeo-emo/master)

The evolutionary algorithms package within the `desdeo` framework.

Currently supported:
* Multi-objective optimization with visualization and interaction support.
* Preference is accepted as a reference point.
* Surrogate modelling (neural networks and genetic trees) evolved via EAs.
* Surrogate assisted optimization

Currently _NOT_ supported:
* Constraint handling

The documentation is currently being worked upon

To test the code, open the [binder link](https://mybinder.org/v2/gh/industrial-optimization-group/desdeo-emo/master) and read example.ipynb.

Read the documentation [here](https://pyrvea.readthedocs.io/en/latest/)

### Requirements:
* Python 3.7 (3.8 is NOT supported)
* [Poetry dependency manager](https://github.com/sdispater/poetry): Only for developers

### Installation process for normal users:
* Create a new virtual enviroment for the project
* Run: `pip install desdeo_emo`

### Installation process for developers:
* Download and extract the code or `git clone`
* Create a new virtual environment for the project
* Run `poetry install` inside the virtual environment shell.

## See the details of various algorithms in the following papers (to be updated)

R. Cheng, Y. Jin, M. Olhofer and B. Sendhoff,
A Reference Vector Guided Evolutionary Algorithm for Many-objective
Optimization, IEEE Transactions on Evolutionary Computation, 2016

The source code of pyrvea is implemented by Bhupinder Saini

If you have any questions about the code, please contact:

Bhupinder Saini: bhupinder.s.saini@jyu.fi\
Project researcher at University of Jyväskylä.