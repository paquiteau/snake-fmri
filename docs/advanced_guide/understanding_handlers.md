---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.15.2
kernelspec:
  display_name: sim
  language: python
  name: sim
---

# Understanding Handlers

In SNAKE-fMRI, handlers are the building block of the simulation. They are configured at initiliasation by a set of parameters, and can then be use to update a simulation object.

Handlers can be chained to create a complex simulation.

## Listing availables handlers

```{code-cell} ipython3
from snkf.handlers import H, list_handlers, HandlerChain
```

```{code-cell} ipython3
list_handlers()
```

## Creating Handlers

Creating handler can be done by getting them through the `H` mapping, and then instantiate them with the parameters you want.

```{code-cell} ipython3
h = H["identity"]
```

## Chaining Handlers

Handlers can be chained in various ways using the `<<` or `>>` operators, the direction of the bracket showing the flow of the simulation data.

Chaining two handlers together would create a `HandlerChain` object. Chaining an Handler and a HandlerChain will add the Handler to the chain (at the end or at the beginning depending on the order). Chaining two `HandlerChain` will create a new one with their concatenated chains.

```{code-cell} ipython3
H1 = H["identity"]()
H2 = H["identity"]()
H3 = H["identity"]()

HC = H1 >> H2 >> H3
HC
```

The following notation are equivalents:

```{code-cell} ipython3
HC1 =  H1 >> H2 >> H3 # to the right
HC2 = H3 << H2 << H1 # to the left
HC3 = H2 >> H3 << H1 # Highly unrecommended ! 

print(HC1)
# Equality is not identity.
print(HC1 == HC2 == HC3)
print(HC1 is HC2)
```

::::{note}
The [precedence rules](https://docs.python.org/3/reference/expressions.html#operator-precedence) of python applies, so the `>>` and `<<` operators are consumed from left to right.
::::

+++

## Exporting and Importing Handlers configs

Handlers config can be written and read from a yaml configuration. This configuration is also used for the runner using OmegaConf/Hydra Configuration files.

```{code-cell} ipython3
HCnew, _ = HandlerChain.from_yaml(HC1.to_yaml())
HCnew 
```
