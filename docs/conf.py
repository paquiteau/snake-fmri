# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "snake-fmri"
copyright = "2023, Pierre-Antoine Comby"
author = "Pierre-Antoine Comby"

master_doc = "index"

from simfmri import __version__ as version  # noqa

# The full version, including alpha/beta/rc tags.
release = version

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "myst_nb",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_copy_source = True

html_last_updated_fmt = ""

html_theme_options = {
    "path_to_docs": "docs",
    "repository_branch": "master",
    "repository_url": "https://github.com/paquiteau/simfmri",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com/",
        "deepnote_url": "https://deepnote.com/",
        "notebook_interface": "jupyterlab",
        "thebe": True,
        # "jupyterhub_url": "https://datahub.berkeley.edu",  # For testing
    },
    "logo": {
        "image_dark": "_static/logo-snake_dark.svg",
        # "text": html_title,  # Uncomment to try text with logo
    },
}
html_logo = "_static/logo-snake_light.svg"
html_favicon = "_static/logo-snake_square.svg"
html_title = "Snake-fmri"

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/devdocs", None),
    "scipy": ("https://scipy.github.io/devdocs", None),
    "matplotlib": ("https://matplotlib.org", None),
}
intersphinx_timeout = 5

suppress_warnings = ["myst.domains", "ref.ref"]

myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    # "html_admonition",
    "html_image",
    "colon_fence",
    # "smartquotes",
    # "replacements",
    # "linkify",
    # "substitution",
]
