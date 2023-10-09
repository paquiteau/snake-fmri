(sim_framework)=

# Simulation Framework

## Modular Approach 

Simfmri has a modular approach for the simulation of fMRI Data, leveraging object-oriented programming.
A core simulation object will be modified and enriched by so-called *handlers* each responsible for a specific part of the model (for instance, defining the anatomical base volume from a phantom or adding noise to a time series). 


Those handlers can be chained to produce complex behaviors from simple operations. See {doc}`../advanced_guide/understanding_handlers` for more details.

:::{tip} 
The combination of a :meth:`SimData` object and handlers can be interpreted as a state machine. 
:::

Defining your handler is also made easy, as shown in {doc}`../advanced_guide/custom_handler.md`. 
Once the simulation has been done, It can be passed down to a reconstructor and the results can then undergo a rudimentary analysis. 

<!-- #+name: block-chain -->
<!-- #+caption: Modular design of the simulator. Each handler manages a particular aspect of the fMRI signal acquisition. TO IMPROVE. -->
<!-- #+attr_latex: :float multicolumn -->
<!-- [[./figs/handlers_chain.pdf]] -->

## Acquisition Model


One of the critical aspects of SNAKE-fMRI is the generation of k-space data.
Yet the discrete approach of the Kspace acquisition, providing a finite volume at a specific time resolution, is incompatible with the continuous and analogous behavior of the neuronal activity.
We thus dissociate the simulation time resolution from the acquisition time resolution based on the following:

During a real scanner acquisition, as the acquisition of k-space data proceeds, the brain signals are also evolving, and the process of spin relaxations is also at play.
Modeling all these aspects would be computationally prohibitive, and we thus propose a simplified approach where the signal is considered constant during the acquisition of a k-space shot, neglecting the relaxation aspect of MR physics.
This approximation is also made at the reconstruction step, allowing one to use the Fourier Transform as a Forward Model of the MRI Acquisition.



:::{figure-md} fig-acq
<img src="images/acquisition2.svg" alt="acquisition" width="70%">

*Acquisition setup*: The high resolution simulation is sampled per shot to create a full kspace frame. Note that the displayed case is overly simplified (2D only, and 7 shot per kspace frame)
:::

The characteristic time for a shot in a T2* MR Sequence used for fMRI is in the order of 25 to 50 ms, which is well below the period of significant physiological signals such as breathing and heartbeat (roughly 1s) or the hemodynamic response (20 to 30s).
Thus, simulating and acquiring a complete volume for each shot is already a faithful approximation of the acquisition process in fMRI.
In practice, the simulation resolution time can also be relaxed as a multiple of a few shot times (for instance, a simulation time of 100 ms for four shots of 25 ms), and the shots are acquired simultaneously, reducing the memory and computational cost of the simulation. The acquisition process is represented in {figure-md:numref}`fig-acq`

