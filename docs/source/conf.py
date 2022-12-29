# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

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
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# generate autosummary even if no references

# mock the import of pysap-fmri
autodoc_mock_imports = ["fmri"]
autodoc_default_options = {"exclude-members": "_abc_impl"}
autosummary_generate = True
napoleon_include_private_with_doc = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_context = {
    # ...
    "default_mode": "light"
}
