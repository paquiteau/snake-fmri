# ⚙️ SNAKE Internals
In this document we present the main principles driving the development of SNAKE. 

## General architecture
SNAKE is built around a modular [core]() that consist of several objects that defines the MRI acquisition system. 
There is also a [toolkit]() that built around the core to provide extra functionalities such as a CLI interface, as well as reconstruction, statistical analysis and plotting tools.

### Main data structures in SNAKE
Here we describe in more details the main data structures that are used in SNAKE.

#### SimConfig

The [](#SimConfig) object holds all the information about the acquisition setup (TR, TE, FA, etc.) as initial setup for the simulation (shape of the phantom, duration of the simulation, etc.)


#### Phantom

The [](#Phantom) represents the phantom state for the simulation, it consists of tissues maps and their MR properties (T1, T2*, PD, etc.). 

It can be generated easily from a known setup using [](#Phantom.from_brainweb) or [](#Phantom.from_mri). 

:::{tip}
The phantom object is a static representation of the Phantom, the dynamic information would be added using [handlers](#handler-docs).
:::

(handler-docs)=
#### Handlers

[`Handlers`](#AbstractHandler) augments the simulation state by declaring how the [](#Phantom) should be modified during the simulation.

(sampler-docs)=
#### Samplers
[](#BaseSampler) are objects responsible to parametrize and generate k-space shots for the acquisition. They can be time-varying, or not.

## Acquisition Engine 
The [](#BaseEngine) is what really perform the acquisition of the k-space data in SNAKE. Currently SNAKE support two acqusition models: GRE acquisition with or without $T_2^*$ and they optionally can be restricted to 2D imaging. See [](#signal-model) for more details.

The base algorithm for the acquisition could be summarized as follows:
```python
for t in range(shot_times):
    shot = get_shot(t) # get the k-space trajectory to acquire at time t
    updated_phantom = base_phantom.copy()
    for h in handlers: # apply all the handlers to the phantom
        updated_phantom = h(updated_phantom, t)
    kspace_data[t] = fourier_model(updated_phantom, shot) # acquire the k-space data

```

(signal-model)=
### Signal Model

The signal model is properly defined in the SNAKE paper [1]. Here we summarize the main equations that are used to generate the k-space data.

TODO


### The MRD file as interface 
Since the acquisition of a shot can be fully determined from the instant it is run and the phantom state, we can save all this information for later use and parallel computation. This is done using the MRD file format. 

:::{tip}
The MRD (or ismrmrd) format is a standard format for storing MRI raw k-space data. It basically consist in a HDF5 file with a dedicated header and a specific structure. 
:::

### Efficient computations with Multiprocessing

to perform the acquisition SNAKE efficiently use the multiprocessing python module to parallelize the acquisition of the k-space data. See [](#snake.engine.BaseEngine.__call__) for more details.


:::{warning}
Currently the SNAKE multiprocessing acquisition and the use of GPU-based NUFFT are incompatible on Windows.

:::
