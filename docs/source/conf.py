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
copyright = "2021 Meerkat"
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
    "recommonmark",
    "sphinx_panels",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "jupyter_sphinx",
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
html_theme = "pydata_sphinx_theme"
html_logo = "../assets/meerkat_banner_padded.svg"
html_favicon = "../assets/meerkat_logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

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
