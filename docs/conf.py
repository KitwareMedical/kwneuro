from __future__ import annotations

import importlib.metadata

project = "kwneuro"
copyright = "2024, Kitware"
author = "Ebrahim Ebrahim, Sadhana Ravikumar, David Allemang"
version = release = importlib.metadata.version("kwneuro")

extensions = [
    "autoapi.extension",
    "myst_parser",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_click",
]

autoapi_dirs = ["../src/kwneuro"]
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = "furo"

myst_enable_extensions = [
    "colon_fence",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]
