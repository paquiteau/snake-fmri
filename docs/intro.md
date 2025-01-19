# 🚀 Getting Started

:::{danger}
These instruction are still WIP
:::
## Installing SNAKE 


:::{attention}
For best performances we recommend running SNAKE on a Linux Machine

:::

```bash
pip install snake-fmri 
```

SNAKE-fMRI depends on [`MRI-NUFFT`](https://github.com/mind-inria/mri-nufft) for the non-Cartesian acquisition. Installing MRI-NUFFT also requires a NUFFT computation backend such as finufft (CPU only) cufinufft or gpuNUFFT (Nvidia GPUs)

:::{tip}
We recommend to use any of the three mentioned NUFFT backend, if you can't decide by yourself, here is a small decision chart:


:::

:::{note} 
Refer to the specific installations guidelines of each backend for more details. 
:::

