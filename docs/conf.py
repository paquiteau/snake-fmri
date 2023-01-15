# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from pathlib import Path

# find project.
sys.path.insert(0, str(Path(__file__).parents[1]))

project = "simfmri"
copyright = "2022, Pierre-Antoine Comby"
author = "Pierre-Antoine Comby"
release = "0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",  # show build time for every doc
    "sphinx.ext.autodoc",  # generate API ref from the code
    "sphinx.ext.autosummary",  # generate API for module
    "sphinx.ext.doctest",  # test snippet in doc
    "sphinx.ext.intersphinx",  # link  with scipy and numpy
    "sphinx.ext.mathjax",  # LaTeX support
    "sphinx.ext.viewcode",  # link to source code
    "sphinx.ext.napoleon",  # numpy docstring support
    "sphinx_gallery.gen_gallery",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# generate autosummary even if no references
autosummary_generate = True

# ignore abc stuff
autodoc_default_options = {"exclude-members": "_abc_impl"}
autodoc_typehints = "both"
napoleon_include_private_with_doc = True


# -- Options for Sphinx Gallery ----------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": ["../examples/"],
    "filename_pattern": "/example_",
    "ignore_pattern": r"/(__init__|conftest)\.py",
}


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "pydata_sphinx_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_context = {
    # ...
    "default_mode": "light"
}
