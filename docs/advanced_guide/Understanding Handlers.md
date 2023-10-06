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

::: {#3ed1bfef .cell .markdown}
# Understanding Handlers

In SNAKE-fMRI, handlers are the building block of the simulation. They are configured at initiliasation by a set of parameters, and can then be use to update a simulation object.

Handlers can be chained to create a complex simulation.

## Listing availables handlers
:::

::: {#83d3056b .cell .code}
``` python
from simfmri.handlers import H, list_handlers, HandlerChain
```
:::

::: {#1d60d456 .cell .code}
``` python
list_handlers()
```
:::

::: {#4083a8e1 .cell .markdown}
## Creating Handlers

Creating handler can be done by calling the `H` function, potentially with arguments,
:::

::: {#1527516e .cell .code}
``` python
```
:::

::: {#341bd1de .cell .markdown}
## Viewing, Exporting Handlers
:::

::: {#e69a49d7 .cell .code}
``` python
h = H["identity"]()

print(h.to_yaml())
```
:::

::: {#89f16831 .cell .markdown}
## Chaining Handlers

Handlers can be chained in various ways using the `<<` or `>>` operators, the direction of the bracket showing the flow of the simulation data.

Chaining two handlers together would create a `HandlerChain` object. Chaining an Handler and a HandlerChain will add the Handler to the chain (at the end or at the beginning depending on the order). Chaining two `HandlerChain` will create a new one with their concatenated chains.
:::

::: {#7c40db38 .cell .code}
``` python

H1 = H["identity"]()
H2 = H["identity"]()
H3 = H["identity"]()

HC = H1 >> H2 >> H3
HC
```
:::

::: {#5f40a4fa .cell .markdown}
The following notation are equivalents:
:::

::: {#3bd23656 .cell .code}
``` python
HC1 =  H1 >> H2 >> H3 # to the right
HC2 = H3 << H2 << H1 # to the left
HC3 = H2 >> H3 << H1 # Highly unrecommended ! 

print(HC1)
print(HC1 == HC2 == HC3)
```
:::

::: {#3f179e5f .cell .markdown}
`` {note} The [precedence rules](https://docs.python.org/3/reference/expressions.html#operator-precedence) of python applies, so the `>>` and `<<` are consumed from left to right. ``
:::

::: {#72ddbd11 .cell .markdown}
## Exporting and Importing Handlers configs

Handlers config can be written and read from a yaml configuration. This configuration is also used for the runner using OmegaConf/Hydra Configuration files.
:::

::: {#d25df440 .cell .code}
``` python
HCnew, _ = HandlerChain.from_yaml(HC1.to_yaml())
```
:::

::: {#9be34662 .cell .code}
``` python
HCnew 
```
:::

::: {#005bbe59 .cell .code}
``` python
d = {"a": 1}
list(d.items())[0][0]
```
:::

::: {#8399421b .cell .code}
``` python
```
:::