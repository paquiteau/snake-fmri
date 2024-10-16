# MRD file for simulation

SNAKE relies on the `.mrd` file format to store the raw data from MRI experiments.

The `.mrd` file format is a standard for sharing raw data of MRI experiments. 
The format is based on the HDF5 file format, which is a widely used file format for scientific data. 
The `.mrd` file format is designed to store raw data from MRI experiments, including the raw k-space data, the acquisition parameters, and other relevant information [^1]
The format is designed to be flexible and extensible, allowing for the storage of a wide range of data types and structures. 

[^1]: https://ismrmrd.readthedocs.io/en/latest/index.html 

## SNAKE MRD file structure 

SNAKE leverages the flexibility of the `.mrd` file format to store all the information need to simulate (f)MRI experiments. 
This allows the simulation to be performed in parallel (and very fast). 

The SNAKE MRD file structure is as follows: 

 - `header` group: contains the metadata for the file, as well as serialized information for the simulation. 
 - `acquisition` group: contains the raw k-space data and the acquisition parameters. 
 - `images` group: contains the static information for the simulation (phantom, coil sensitivity maps, etc.).
 - `waveforms` group: contains the dynamic information for the simulation (motion, physiological noise, etc.).
 
 
## Reading and writing MRD files

SNAKE provides a set of functions to read and write MRD files in the `snake.mrd_utils` module, built on top of the [`ismrmrd`](https://github.com/ismrmrd/ismrmrd-python) library. 




 


