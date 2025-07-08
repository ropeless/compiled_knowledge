import os
import pydoc
import shutil
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import toml

BUILD_API_DOC: bool = True
FORCE_REBUILD: bool = True
INSTANTIATE_TEMPLATES: bool = True
EXECUTE_NOTEBOOKS: bool = True
OPEN_DOCUMENT_HTML: bool = True


def main() -> None:
    """
    Build the documentation.

    This script will explicitly update the document files by instantiating
    templates and executing notebooks in-place.

    The updated document files should be checked and committed to the repository as
    they will be used directly by Read The Docs when pushed to GitHub.
    """
    project_dir: Path = Path(__file__).parent
    src_dir: Path = project_dir / 'src'
    ck_package_dir: Path = src_dir / 'ck'
    docs_dir: Path = project_dir / 'docs'
    build_dir: Path = docs_dir / '_build'
    html_index = build_dir / 'html/index.html'
    api_docs_dir = docs_dir / 'api'
    api_docs_output_dir = api_docs_dir / 'build'

    if BUILD_API_DOC:
        # TODO - not working yet

        # Ensure `_static` directory exists
        static: Path = api_docs_dir / '_static'
        if not static.exists():
            static.mkdir()

        # Generate RST files
        cmd: List[str] = ['sphinx-apidoc', '-o',  api_docs_output_dir.as_posix(), ck_package_dir.as_posix()]
        subprocess.run(cmd, capture_output=False, check=True)

        # Generate HTML files
        cmd: List[str] = ['sphinx-build', '-b',  'html', api_docs_dir.as_posix(), api_docs_output_dir.as_posix()]
        subprocess.run(cmd, capture_output=False, check=True)

    # TODO Disabled for now while debugging API documentation generation process.
    #
    # if INSTANTIATE_TEMPLATES:
    #     instantiate_templates(project_dir, docs_dir)
    #
    # if FORCE_REBUILD and build_dir.exists():
    #     shutil.rmtree(build_dir)
    #
    # if EXECUTE_NOTEBOOKS:
    #     for notebook_path in docs_dir.glob('*.ipynb'):
    #         execute_notebook(notebook_path)
    #
    # run_jupyter_book()
    #
    # if OPEN_DOCUMENT_HTML:
    #     webbrowser.open(html_index.as_uri())


def run_jupyter_book() -> None:
    """
    Run the Jupyter Book command to build the documentation.
    """
    cmd: List[str] = ['jupyter-book', 'build', '-qq', 'docs']
    subprocess.run(cmd, capture_output=False, check=True)


def load_pyproject(file_path: Path) -> Dict[str, Any]:
    """
    Loads the pyproject.toml file as a nested dictionary.

    Args:
        file_path: Path to the pyproject.toml file.

    Returns:
        the toml dictionary
    """
    with open(file_path, 'r') as f:
        return toml.load(f)


def instantiate_templates(project_dir: Path, docs_dir: Path) -> None:
    """
    Instantiate Markdown documents from found templates.

    Args:
        project_dir: where to find the 'pyproject.toml' file.
        docs_dir: where to find the documents files.
    """
    pyproject = load_pyproject(project_dir / 'pyproject.toml')
    # These values will be inserted into the template using `str.format`.

    fields: Dict[str, str] = {
        'version': pyproject['project']['version'],
        'version_note': pyproject.get('doc_extra', {}).get('version_note', ''),
        'date': datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S (%Z)'),
    }

    for template_file in docs_dir.glob('*_template.md'):
        dest_name = template_file.name[:-len('_template.md')] + '.md'
        dest_file: Path = docs_dir / dest_name

        with open(template_file, 'r') as f:
            lines = [
                line.format(**fields)
                for line in f.readlines()
            ]

        with open(dest_file, 'w') as f:
            f.writelines(lines)


def execute_notebook(notebook_path: Path) -> None:
    """
    Executes a Jupyter notebook and save the output inplace.

    Args:
        notebook_path: Path to the Jupyter notebook file.
    """
    print(f'Executing Jupyter notebook {notebook_path.name}')

    # Construct the command to execute the notebook using nbconvert
    command = [
        sys.executable,
        '-m',
        'nbconvert',
        '--to', 'notebook',
        '--execute',
        '--allow-errors',
        '--log-level', 'ERROR',
        '--inplace',
        notebook_path,
    ]

    # Execute the command
    subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
