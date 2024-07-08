import os
import sys
sys.path.insert(0, os.path.abspath('../../src/'))

from mili import __version__

import sphinx_rtd_theme

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

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme"
]

html_theme = 'sphinx_rtd_theme'