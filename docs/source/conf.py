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
import sphinx_rtd_theme
sys.path.insert(0, os.path.split(os.path.abspath(".."))[0])

# -- Project information -----------------------------------------------------

project = '6DOF Rocket Trajectory Simulation'
copyright = '2020, Jago Strong-Wright and Daniel Gibbons'
author = 'Jago Strong-Wright and Daniel Gibbons'

# The full version, including alpha/beta/rc tags
release = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "rst2pdf.pdfbuilder"
]

#html_theme = "sphinx_rtd_theme"
html_theme = "pydata_sphinx_theme"
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

pdf_documents = [('index', u'rst2pdf', u'Trajectory Simulation Package', u'Jago Strong-Wright & Daniel Gibbons'),]
