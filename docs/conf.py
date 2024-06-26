###############################################################################
# Auto-generated by `jupyter-book config`
# If you wish to continue using _config.yml, make edits to that file and
# re-generate this one.
###############################################################################
add_module_names = False
author = "Pierre-Antoine Comby"
autoclass_content = "both"
autodoc_default_options = {"ignore-module-all": True}
autodoc_member_order = "bysource"
autodoc_mock_imports = ["fmri"]
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "label"
comments_config = {"hypothesis": False, "utterances": False}
copyright = "2022"
exclude_patterns = ["_build", "_templates"]
extensions = [
    "sphinx_togglebutton",
    "sphinx_copybutton",
    "myst_nb",
    "jupyter_book",
    "sphinx_thebe",
    "sphinx_comments",
    "sphinx_external_toc",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_book_theme",
    "numpydoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinxcontrib.bibtex",
    "sphinx_jupyterbook_latex",
]
external_toc_exclude_missing = False
external_toc_path = "_toc.yml"
html_baseurl = ""
html_favicon = ""
html_logo = "images/logos/logo-snake_light.svg"
html_sourcelink_suffix = ""
html_theme = "sphinx_book_theme"
html_theme_options = {
    "search_bar_text": "Search this book...",
    "launch_buttons": {
        "notebook_interface": "classic",
        "binderhub_url": "",
        "jupyterhub_url": "",
        "thebe": False,
        "colab_url": "",
    },
    "path_to_docs": "docs",
    "repository_url": "https://github.com/executablebooks/jupyter-book",
    "repository_branch": "master",
    "extra_footer": "",
    "home_page_in_toc": True,
    "announcement": "",
    "analytics": {"google_analytics_id": ""},
    "use_repository_button": True,
    "use_edit_page_button": False,
    "use_issues_button": True,
}
html_title = "SNAKE-fMRI Documentation"
intersphinx_mapping = {
    "python": ["https://docs.python.org/3/", None],
    "matplotlib": ["https://matplotlib.org/stable", None],
    "numpy": ["https://numpy.org/doc/stable/", None],
    "pandas": ["https://pandas.pydata.org/pandas-docs/stable", None],
}
latex_engine = "pdflatex"
myst_enable_extensions = ["colon_fence", "deflist"]
myst_url_schemes = ["mailto", "http", "https"]
nb_execution_allow_errors = False
nb_execution_cache_path = ""
nb_execution_excludepatterns = []
nb_execution_in_temp = False
nb_execution_mode = "force"
nb_execution_timeout = 30
nb_kernel_rgx_aliases = {".*": "python3"}
nb_output_stderr = "show"
numfig = True
numpydoc_show_class_members = False
pygments_style = "sphinx"
suppress_warnings = ["myst.domains"]
use_jupyterbook_latex = True
use_multitoc_numbering = True
