# SNAKE-fMRI: A Simulation and Reconstruction Framework for fMRI

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
pip install git+github.com/pysap-fmri
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
The documentation is available at https://snake-fmri.readthedocs.io/en/latest/, our [preprint](XXXX) describe also the framework in details.

Don't hesitate to also check the [examples gallery](XXXXX) .

# Running simulation and benchmarks 
## Available commands 

3 CLI interfaces are able to use the configuration folder to perform differents task: 
 - `snkf-main` to do a full simulation + reconstruction + validation 
 

## Configurations Files
The configuration  are located in `snkf/conf` and articulates over 3 main componenents: 
- the simulation
- the reconstruction
- the validation via statistical analysis



# Citing SNAKE-fMRI
If you use SNAKE-fMRI in your research, please cite the following paper:

> 
> 

```
@article{snakefmri,
}
```

# License
SNAKE-fMRI is licensed under the MIT License. See the LICENSE file for more information.
