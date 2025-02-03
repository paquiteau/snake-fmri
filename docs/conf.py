"""
Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
from pathlib import Path
import sys
from myst_sphinx_gallery import GalleryConfig, generate_gallery

sys.path.insert(0, os.path.abspath("../.."))  # Source code dir relative to this file
sys.path.insert(1, os.path.abspath("_ext/"))  # load custom extensions

# -- Project information -----------------------------------------------------

project = "SNAKE-fMRI"
copyright = "2022, SNAKE-fMRI Contributors"
author = "SNAKE-fMRI Contributors"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_tippy",
    "sphinx_copybutton",
    "autodoc2",
    "myst_sphinx_gallery",
    "myst_nb",
    "scenario",
    "sphinx_gallery.gen_gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


autodoc2_packages = ["../src/snake/"]
autodoc2_output_dir = "auto_api"
autodoc2_docstring_parser_regexes = [
    (
        ".*",
        "numpy_docstring",
    ),
]
pygments_style = "sphinx"
highlight_language = "python"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}


# -- MyST configuration ---------------------------------------------------
#
# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html
myst_enable_extensions = [
    "smartquotes",  # use “ ”
    "replacements",  # (c) and (tm) etc
    "dollarmath",  # use $ and $$ for maths. (more friendly than ```{math}...``` or {math}`...`)
    "amsmath",  # direct support for \begin{equation}
    "linkify",  # automatic link marking
    "colon_fence",  # use colon for directive (but please keep using ``` for maths and codes)
    "html_admonition",  # can use <div class="admonition note">
]

# nb_custom_formats = {
#     ".yaml": "scenario.scenario2file",
#     ".py": ("jupytext.reads", {"fmt": "py:percent"}),
# }

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

# The scenario Gallery is built using myst-sphinx-gallery.
# # Gallery of examples
# myst_sphinx_gallery_config = GalleryConfig(
#     examples_dirs="../examples",
#     gallery_dirs="auto_examples",
#     root_dir=Path(__file__).parent,
#     notebook_thumbnail_strategy="code",
#     thumbnail_strategy="last",
# )


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/paquiteau/snake-fmri/",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "home_page_in_toc": True,
    "logo": {
        "image_light": "_static/logos/snake-fmriV2-logo.png",
        "image_dark": "_static/logos/snake-fmriV2-logo_dark.png",
    },
}
html_title = "SNAKE-fMRI Documentation"


# ----- Sphinx Gallery options ----------------------------------
#

sphinx_gallery_conf = {
    "examples_dirs": "../examples",  # path to your example scripts
    "gallery_dirs": "auto_examples",  # path to where to save gallery generated output
}
