<img align="left" width="33%" src="https://github.com/paquiteau/snake-fmri/blob/main/docs/images/logos/logo-snake_light.svg"> 
<h1> A Simulator from Neurovascular coupling to Acquisition of K-space data for Exploration of fMRI Technique </br></br></h1>


[![Test](https://github.com/paquiteau/snake-fmri/actions/workflows/test.yml/badge.svg)](https://github.com/paquiteau/snake-fmri/actions/workflows/test.yml)
[![deploy-docs](https://github.com/paquiteau/snake-fmri/actions/workflows/deploy-docs.yml/badge.svg)](https://paquiteau.github.io/snake-fmri)
[![HAL](https://img.shields.io/badge/preprint-04533862-purple?logo=HAL&logoColor=white)](https://hal.science/hal-04533862)

[![python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=blue)](https://pypi.org/project/snake-fmri)
![black](https://img.shields.io/badge/code--style-black-black)
![ruff](https://img.shields.io/badge/lint-ruff-purple?logo=stackblitz&logoColor=yellow)



This package provides a simulation and reconstruction framework for fMRI data. It is designed to be used for benchmarking and testing of fMRI reconstruction methods.


# Installation
## Requirements 
- A working Python 3.10 environment  or higher 
- Optional: a working CUDA environment for NUFFT accelerations.

## Installation
To install SNAKE-fMRI, you can use pip. 

```bash
pip install snake-fmri 
# Required for the reconstruction 
pip install git+https://github.com/paquiteau/pysap-fmri
# Recommended for the nufft acceleration 
pip install gpunufft # or cufinufft 
```

Or the latest version from the repository:

```bash
git clone git@github.com/paquiteau/snake-fmri 
cd snake-fmri 
pip install -e .
```
After installation Snake-fMRI is available as the `snkf` module: 

``` python
import snkf

```

# Documentation
The documentation is available at https://paquiteau.github.io/snake-fmri/, our [preprint](https://hal.science/hal-04533862) describe also the framework in details.

Don't hesitate to also check the [examples gallery (TBA)](https://paquiteau.github.io/snake-fmri).

# Running simulation and benchmarks 
## Available commands 

3 CLI interfaces are able to use the configuration folder to perform differents task: 
 - `snkf-main` to do a full simulation + reconstruction + validation 
 

## Configurations Files
The configuration  are located in `snkf/conf` and articulates over 3 main components: 
- the simulation
- the reconstruction
- the validation via statistical analysis



# Citing SNAKE-fMRI
If you use SNAKE-fMRI in your research, please cite the following paper:

>  Pierre-Antoine Comby, Alexandre Vignaud, Philippe Ciuciu. SNAKE-fMRI: A modular fMRI data simulator from the space-time domain to k-space and back. 2024. ⟨hal-04533862⟩ 

```bibtex
@unpublished{comby:hal-04533862,
  TITLE = {{SNAKE-fMRI: A modular fMRI data simulator from the space-time domain to k-space and back}},
  AUTHOR = {Comby, Pierre-Antoine and Vignaud, Alexandre and Ciuciu, Philippe},
  URL = {https://hal.science/hal-04533862},
  NOTE = {working paper or preprint},
  YEAR = {2024},
  MONTH = Mar,
  KEYWORDS = {fMRI ; Brain Imaging ; Accelerated sampling ; Compressed Sensing ; Simulation ; Open Source ; Python ; np ndarray , np newaxis] ; np ; ndarray , np ; newaxis]},
  PDF = {https://hal.science/hal-04533862/file/main.pdf},
  HAL_ID = {hal-04533862},
  HAL_VERSION = {v1},
}
```

# License

SNAKE-fMRI is licensed under the MIT License. See the LICENSE file for more information.
