# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'jwstnoobfriend'
copyright = '2024, Zhu Chenghao'
author = 'Zhu Chenghao'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'numpydoc',  # Use numpydoc instead of napoleon
]

templates_path = ['_templates']
exclude_patterns = []

napoleon_numpy_docstring = True
napoleon_google_docstring = False
autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'private-members': False,
    'special-members': False,
    'inherited-members': True,
    'show-inheritance': True,
}
autoclass_content = 'class'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
import os
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))
