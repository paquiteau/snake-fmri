---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: pandoc
      format_version: 3.1.8
      jupytext_version: 1.15.2
  kernelspec:
    display_name: sim
    language: python
    name: sim
  nbformat: 4
  nbformat_minor: 5
---

::: {#8bc27188 .cell .markdown}
# Understanding Handlers

In SNAKE-fMRI, handlers are the building block of the simulation. They are configured at initiliasation by a set of parameters, and can then be use to update a simulation object.

Handlers can be chained to create a complex simulation.

## Listing availables handlers
:::

::: {#bd2cb6e0 .cell .code}
``` python
from simfmri.handlers import H, list_handlers, HandlerChain
```
:::

::: {#d7ccec90 .cell .code}
``` python
list_handlers()
```
:::

::: {#0d6f2830 .cell .markdown}
## Creating Handlers

Creating handler can be done by getting them through the `H` mapping, and then instantiate them with the parameters you want.
:::

::: {#ac349707 .cell .code}
``` python
h = H["identity"]
```
:::

::: {#489e704b .cell .markdown}
## Chaining Handlers

Handlers can be chained in various ways using the `<<` or `>>` operators, the direction of the bracket showing the flow of the simulation data.

Chaining two handlers together would create a `HandlerChain` object. Chaining an Handler and a HandlerChain will add the Handler to the chain (at the end or at the beginning depending on the order). Chaining two `HandlerChain` will create a new one with their concatenated chains.
:::

::: {#62415320 .cell .code}
``` python

H1 = H["identity"]()
H2 = H["identity"]()
H3 = H["identity"]()

HC = H1 >> H2 >> H3
HC
```
:::

::: {#eef70860 .cell .markdown}
The following notation are equivalents:
:::

::: {#a45f9841 .cell .code}
``` python
HC1 =  H1 >> H2 >> H3 # to the right
HC2 = H3 << H2 << H1 # to the left
HC3 = H2 >> H3 << H1 # Highly unrecommended ! 

print(HC1)
# Equality is not identity.
print(HC1 == HC2 == HC3)
print(HC1 is HC2)
```
:::

::: {#f51ea8d6 .cell .markdown}
`` {note} The [precedence rules](https://docs.python.org/3/reference/expressions.html#operator-precedence) of python applies, so the `>>` and `<<` operators are consumed from left to right. ``
:::

::: {#c6f81182 .cell .markdown}
## Exporting and Importing Handlers configs

Handlers config can be written and read from a yaml configuration. This configuration is also used for the runner using OmegaConf/Hydra Configuration files.
:::

::: {#bd2b64b6 .cell .code}
``` python
HCnew, _ = HandlerChain.from_yaml(HC1.to_yaml())
HCnew 
```
:::

::: {#826e109a .cell .code}
``` python
```
:::

::: {#f80a034d .cell .code}
``` python
HCnew 
```
:::