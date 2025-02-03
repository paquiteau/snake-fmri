# ❔ Frequently Asked Questions


## What does SNAKE mean ? 

- Simulator from Neuro-Activation to K-space Evaluation 
- Simulator of New Acquisition for K-space Exploration
- SNAKE is Not A K-space Emulator 


## Is it SNAKE or SNAKE-fMRI ? 

It's SNAKE, just SNAKE. the development of this package started as SNAKE-fMRI, and still target mainly functional MRI application, but it is not limited to functional MRI. It can be used in anatomical settings as long as you stick to GRE setting ([Spin Echo sequences are not supported](#spin-echo-missing)

## How does SNAKE works ?

See [](snake-internal.md) and the [](../auto_api/index.rst) if you need more information, or go dive into the [source code](https://github.com/paquiteau/snake-fmri)


## Why do you don't you  model ... ? 
### ... field inhomogeneities

There is a detailed explanation in our [paper](#paper) (see section 7.4.1) but in a nutshell, adding field inhomogeneities is very expensive to add in term of computations, the ideal acquisition model of SNAKE is already a challenging settings for fMRI data acquisition and image reconstruction.

### ... eddy current artifacts on trajectories
### ... 

## I want to add this awesome feature to SNAKE, what should I do ?
Well, first try it ! SNAKE is designed to be extensible and modular, so you should be able to add specific [handlers](#handlers) or [samplers](#samplers). If you wish to share your improvement of SNAKE, please open a [Pull Request](https://github.com/paquiteau/snake-fmri/pulls)

## How can I share my simulation with someone else ? 

IF you have a standard configuration file (e.g. ``scenario.yaml``) you could just send that, and let the other party run the simulation on their side. Or you could decide to share the (heavier) `.mrd` file resulting from the acquisition simulation. Internally it should contains all the information required to describe the scenario. 

## Does SNAKE supports ... 
### ... Multi-Echo GRE fMRI ? 

No. But, there are some workaround possibles: You can run several acquisition with different echo time (using the same parametrization everywhere), getting an MRD file for each echo spacing. Then you can combine them manually to perform your reconstruction. If you have an example of this mechanism, your contribution is welcome !

(spin-echo-missing)=
### ... Spin-Echo Acquisition ? 
No. fMRI is overwehlmingly done using Gradient Recall Echo sequences, and the 


