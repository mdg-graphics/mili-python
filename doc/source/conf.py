from mili import __version__

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Mili Python'
copyright = '2023, Lawrence Livermore National Security, LLC'
author = 'William Tobin, Ryan Hathaway, Kevin Durrenberger'
release = '.'.join(map(str,__version__))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = []
exclude_patterns = []


# Markdown configuration
extensions = ["myst_parser"]
myst_heading_anchors=2

# Latex configuration
latex_elements = {
    "extraclassoptions": "openany,oneside",
    "papersize": "a4paper",
    "pointsize": "10pt",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = []
