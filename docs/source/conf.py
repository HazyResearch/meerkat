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
import os
import sys
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path("../../meerkat/version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

sys.path.insert(0, os.path.abspath(""))
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../.."))
sys.setrecursionlimit(1500)

# -- Project information -----------------------------------------------------

project = "Meerkat"
copyright = "2022 Meerkat"
author = "The Meerkat Team"

# The full version, including alpha/beta/rc tags
# release = "0.0.0dev"
version = release = main_ns["__version__"]

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx_rtd_theme",
    "nbsphinx",
    # "recommonmark",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "myst_nb",
    "sphinx.ext.intersphinx",
    "sphinx_panels",
    "sphinx_book_theme",
    "sphinx_external_toc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_logo = "../assets/meerkat_banner_padded.svg"
html_favicon = "../assets/meerkat_logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

# From jupyter-book default sphinx config
# https://github.com/executablebooks/jupyter-book/blob/421f6198728b21c94726a10b61776fb4cc097d72/jupyter_book/config.py#L23
html_add_permalinks = "¶"
html_sourcelink_suffix = ""
numfig = True
panels_add_bootstrap_css = False

# Don't show module names in front of class names.
add_module_names = False

# Sort members by group
autodoc_member_order = "groupwise"

# Color Scheme
panels_css_variables = {
    "tabs-color-label-active": "rgb(108,72,232)",
    "tabs-color-label-inactive": "rgba(108,72,232,0.5)",
}

todo_include_todos = True

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
    ".md": "myst-nb",
}

external_toc_path = "_toc.yml"

html_theme_options = {
    "repository_url": "https://github.com/robustness-gym/meerkat/",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "doc/source",
    "home_page_in_toc": False,
    "show_navbar_depth": 1,
    "launch_buttons": {
        "notebook_interface": "jupyterlab",
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
    },
    "announcement": "<div class='topnav'></div>",
}
