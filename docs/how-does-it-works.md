# How does SNAKE work ?


SNAKE is a modular Simulator (f)MRI data. 
A simulation is described by a `SimConfig` object, a `Phantom` and a set of `Handlers`. 
All of this information is then embedded in a `.mrd` file (see : [](../mrd-simulation.md)). 
This is then use as source for the `Engine` that is going to fill this file with simulated k-space data.

For an example of minimalist simulation you can check **TODO**

