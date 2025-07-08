import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import toml


project_dir = Path(__file__).parent.parent.parent
src_dir = project_dir / 'src'
sys.path.insert(0, src_dir.as_posix())

with open(project_dir / 'pyproject.toml', 'r') as f:
    pyproject: Dict[str, Any] = toml.load(f)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Compiled Knowledge'
copyright = datetime.now().astimezone().strftime('%Y')
author = pyproject['project']['authors'][0]['name']
version = pyproject['project']['version']
release = pyproject['project']['version']

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.todo', 'sphinx.ext.viewcode', 'sphinx.ext.autodoc']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
