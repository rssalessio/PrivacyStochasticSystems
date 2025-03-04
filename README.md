# PrivacyStochasticSystems Library

Code used in *"Minimizing Information Leakage of Abrupt Changes in Stochastic Systems" - Alessio Russo, Alexandre Proutiere, 2021*.

_Author_: Alessio Russo (PhD Student at KTH - alessior@kth.se)

## License
Our code is released under the MITlicense (refer to the [LICENSE](https://github.com/rssalessio/PoisoningDataDrivenControl/blob/master/LICENSE) file for details).

## Requirements
To use the library you need atleast Python 3.7. To run the notebooks you need to install jupyter notebooks.

Other required dependencies:
- NumPy
- SciPy
- CVXPY
- DCCP
- Matplotlib
- TQDM

If you have Conda installed you may install the required packages using the command ``conda env create --file=env.yaml``

## Usage/Examples

Check the notebook files. If possible, use the MOSEK solver. Alternatively, one can use ECOS (which is included in cvxpy). That may require some fine-tuning.

- [Linear system - full information](https://github.com/rssalessio/PrivacyStochasticSystems/blob/main/linear_system_full_information_case.ipynb)
- [Linear system - limited information](https://github.com/rssalessio/PrivacyStochasticSystems/blob/main/linear_system_limited_information_case.ipynb)
- [MDP with 3 states](https://github.com/rssalessio/PrivacyStochasticSystems/blob/main/mdp_example.ipynb)

## Citations

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

