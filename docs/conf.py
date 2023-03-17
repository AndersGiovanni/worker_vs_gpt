"""Sphinx configuration."""
project = "Worker vs. GPT"
author = "Anders Giovanni Møller & Jacob Aarup Dalsgaard"
copyright = "2023, Anders Giovanni Møller & Jacob Aarup Dalsgaard"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
