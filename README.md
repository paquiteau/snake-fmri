# fMRI data Simulator  and validator

This package provides:
 - An easy to use fMRI data simulator, using phantoms and explicit forward modelling steps (Brain activations, noise, motion, etc.)
 
 - A set of tool to perform a benchmark of fMRI data reconstructors, mostly all implemented in [pysap-fmri](https://github.com/paquiteau/pysap-fmri). 
 
 
 
 

# Running simulation and benchmarks 

simfmri comes with helper tools to configure and run simulation, and as such can be use to perform benchmark of fMRI reconstruction method.  It was developed for this purpose. 


## Available commands 

3 CLI interfaces are able to use the configuration folder to perform differents task: 
 - `snkf-main` to do a full simulation + reconstruction + validation 
 - `snkf-data` to generate a dataset of simulation
 - `snkf-rec` to evaluate reconstructions methods (and do the statistics) over an existing dataset. 
 
 Typically you would create a dataset with `snkf-data` and then run `snkf-rec` on it.
 

## Configurations Files
The configuration  are located in `snkf/runner/conf` and articulates over 3 main componenents: 
- the simulation
- the reconstruction
- the validation via statistical analysis

### Simulation 
located in `snkf/runner/conf/simulation`
### Reconstruction
located in `snkf/runner/conf/reconstruction`

### Validation


# Related packages 
## Dependencies
- https://github.com/mind-inria/mri-nufft : A python package to perform NUFFT in a MRI context.
- https://github.com/paquiteau/pysap-fmri : A python package to perform fMRI reconstruction.
- https://github.com/nilearn/nilearn : A python package to perform fMRI analysis.
- https:

## Other simulation software
 - neuroRsim/fmrisim 
 - POSSUM (from FSL)
 

