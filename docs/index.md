---
sd_hide_title: true
---

# 🔎 Overview 


::::{grid}
:reverse:
:gutter: 3 4 4 4
:margin: 1 2 1 2

:::{grid-item}
:columns: 12 4 4 4

```{image} _static/logos/snake-fmriV2-logo.png
:width: 200px
:class: sd-m-auto only-light
:name: landing-page-logo
```
```{image} _static/logos/snake-fmriV2-logo_dark.png
:width: 200px
:class: sd-m-auto only-dark
:name: landing-page-logo-dark
```
:::

:::{grid-item}
:columns: 12 8 8 8
:child-align: justify
:class: sd-fs-5

```{rubric} SNAKE - A Simulator from Neuro-Activation to K-space Evaluation
```

SNAKE is a **S**imulator from **N**euro-**A**ctivation to **K**-space **E**valuation used to develop and benchmark new acquisition and reconstruction strategies for functional MRI. 

````{div} sd-d-flex-row
```{button-ref} intro
:ref-type: doc
:color: primary
:class: sd-rounded-pill sd-mr-3

Get Started
```
```{button-link} https://arxiv.org/abs/2404.08282v1
:color: success
:class: sd-rounded-pill sd-mr-3

Read the Paper 
```
````

:::
::::



```{rubric} Highlighted Features
```

- Efficient Simulation model 
- Modular and easily extensible 
- End-to-End validation of fMRI acquisition and reconstruction strategies 
- 2D and 3D support at high resolution


<!-- TOC -->

```{toctree}
:hidden:
intro.md

```



```{toctree}
:hidden:
:caption: 📚 Explanations

explanations/snake-internal.md
explanations/faq.md
explanations/cli-interface.md
```

```{toctree}
:hidden:
:caption: 💡Examples

auto_examples/anatomical/index
```

```{toctree}
:hidden:
:caption: 📜 Scenarios configuration 
auto_scenarios/index
```

```{toctree}
:hidden:
:caption: 🛠 API Reference

auto_api/snake/snake.core.rst
auto_api/snake/snake.toolkit.rst
```


```{toctree}
:hidden:
:caption: Miscellaneous

misc/contributors
misc/development
misc/license
```
