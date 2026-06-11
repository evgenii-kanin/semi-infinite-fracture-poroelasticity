# Steadily moving semi-infinite fracture in plane poroelasticity

This repository contains the code and auxiliary materials accompanying the paper **"Steadily moving semi-infinite fracture in plane poroelasticity"**.

The repository provides an implementation of the boundary integral formulation, the asymptotics-informed collocation method, and the verification examples presented in the paper.

## Repository contents

```text
semi-infinite-fracture-poroelasticity/
├── data/
│   ├── precomputed_integrals/
│   └── precomputed_results/
├── exports/
├── mathematica/
│   └── auxiliary_analytical_calculations.nb
├── notebooks/
│   ├── verification_examples.ipynb
│   └── integral_precomputation.ipynb
├── src/
│   ├── __init__.py
│   ├── collocation_utils.py
│   ├── integration_utils.py
│   ├── poroelastic_kernels.py
│   ├── precomputed_integrals.py
│   ├── solvers_shear_fracture.py
│   └── solvers_tensile_fracture.py
├── LICENSE
├── README.md
└── requirements.txt
```

## Description

The code focuses on steadily moving semi-infinite fractures in plane poroelasticity. It includes:

* dimensionless kernel functions for steadily moving fluid sources and edge dislocations;
* numerical routines for precomputing integral matrices used in the collocation method;
* solvers for tensile and shear semi-infinite fracture benchmarks;
* verification examples comparing numerical solutions with analytical and semi-analytical reference solutions;
* auxiliary Mathematica calculations used to generate reference data for selected benchmarks.

## Verification examples

The main verification notebook is:

```text
notebooks/verification_examples.ipynb
```

It reproduces the verification benchmarks presented in the paper, including:

1. semi-infinite tensile fracture with prescribed exponential normal loading and impermeable fracture surfaces;
2. semi-infinite tensile fracture with prescribed exponential normal loading and permeable fracture surfaces;
3. semi-infinite tensile fracture with prescribed exponential pore pressure;
4. semi-infinite shear fracture with uniform shear loading over a finite region.

The notebook uses the precomputed data stored in:

```text
data/precomputed_results
data/precomputed_integrals
```

## Integral matrix precomputation

The notebook

```text
notebooks/integral_precomputation.ipynb
```

demonstrates how the integral matrices used by the collocation solvers are evaluated. It is intended as an explanatory notebook and does not overwrite the precomputed arrays distributed in `data/precomputed_integrals`.

The corresponding numerical integration utilities are implemented in:

```text
src/integration_utils.py
```

## Auxiliary Mathematica calculations

The Mathematica notebook

```text
mathematica/auxiliary_analytical_calculations.nb
```

contains auxiliary analytical calculations used in the verification examples. These include calculations of pore pressure profiles and pressure fields for the Atkinson-Craster tensile fracture benchmarks, as well as the evaluation of the stress intensity factor for the Rice-Simons shear fracture benchmark.

Regenerated data from the Mathematica notebook can be saved to `data/results`, which is not distributed with the repository by default.

These data can be used in the verification notebook instead of the precomputed data stored in:

```text
data/precomputed_results
```

## Static exported files

The folder

```text
exports/
```

contains static exported versions of selected notebooks and auxiliary calculations for convenient viewing without running Python or Mathematica.

## Installation

Clone the repository:

```bash
git clone https://github.com/evgenii-kanin/semi-infinite-fracture-poroelasticity.git
cd semi-infinite-fracture-poroelasticity
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

The required packages are:

```text
numpy
scipy
matplotlib
mpmath
tqdm
jupyter
```

## Running the notebooks

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Then open:

```text
notebooks/verification_examples.ipynb
```

or

```text
notebooks/integral_precomputation.ipynb
```

## Citation

If you use this repository, please cite the associated paper:

```text
Kanin, E., Möri, A., Garagash, D., and Lecampion, B. Steadily moving semi-infinite fracture in plane poroelasticity. arXiv preprint arXiv:2604.18483, 2026.
```

BibTeX:

```bibtex
@article{kanin2026steadily,
  title={Steadily moving semi-infinite fracture in plane poroelasticity},
  author={Kanin, Evgenii and M{\"o}ri, Andreas and Garagash, Dmitry and Lecampion, Brice},
  journal={arXiv preprint arXiv:2604.18483},
  year={2026}
}
```

The citation will be updated with the final journal reference once the article details are available.

## License

This repository is distributed under the BSD 3-Clause License. See the `LICENSE` file for details.
