# ❔ Frequently Asked Questions


## What does SNAKE mean ? 

- Simulator from Neuro-Activation to K-space Evaluation 
- Simulator of New Acquisition for K-space Exploration
- SNAKE is Not A K-space Emulator 


## Is it SNAKE or SNAKE-fMRI ? 

It's SNAKE, just SNAKE. the development of this package started as SNAKE-fMRI, and still target mainly functional MRI application, but it is not limited to functional MRI. It can be used in anatomical context[^1]

[^1]: Who can do more can do less, right ?

## How does SNAKE works ?

## How do I use SNAKE for my work ? 

## Why do you don't model this or that ? 

## I want to add this awesome feature to SNAKE, what should I do ?

## How can I share my simulation with someone else ? 

## Does SNAKE supports ... 
### ... Multi-Echo GRE fMRI ? 

No. But, there are some workaround possibles: You can run several acquisition with different echo time (using the same parametrization everywhere), getting an MRD file for each echo spacing. Then you can combine them manually to perform your reconstruction. If you have an example of this mechanism, your contribution is welcome !

### ... Spin-Echo Acquisition ? 
No. fMRI is overwehlmingly done using Gradient Recall Echo sequences, and the 


