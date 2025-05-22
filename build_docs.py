import shutil
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import List

import toml

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
    docs_dir: Path = project_dir / 'docs'
    build_dir: Path = docs_dir / '_build'
    html_index = build_dir / 'html/index.html'

    if INSTANTIATE_TEMPLATES:
        make_front_matter(project_dir, docs_dir)

    if FORCE_REBUILD and build_dir.exists():
        shutil.rmtree(build_dir)

    if EXECUTE_NOTEBOOKS:
        for notebook_path in docs_dir.glob('*.ipynb'):
            execute_notebook(notebook_path)

    run_jupyter_book()

    if OPEN_DOCUMENT_HTML:
        webbrowser.open(html_index.as_uri())


def run_jupyter_book() -> None:
    """
    Run the Jupyter Book command to build the documentation.
    """
    cmd: List[str] = ['jupyter-book', 'build', '-qq', 'docs']
    subprocess.run(cmd, capture_output=False, check=True)


def get_project_version(file_path: Path) -> str:
    """
    Extracts the project version from a pyproject.toml file.

    Args:
        file_path: Path to the pyproject.toml file.

    Returns:
        str: The project version string
    """
    with open(file_path, 'r') as f:
        data = toml.load(f)
    return data['project']['version']


def make_front_matter(project_dir: Path, docs_dir: Path) -> None:
    """
    Update the "front matter" document form its template.

    Args:
        project_dir: where to find the 'pyproject.toml' file.
        docs_dir: where to find the documents files.
    """
    # These values will be inserted into the template using `str.format`.
    version: str = get_project_version(project_dir / 'pyproject.toml')
    date: str = datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S (%Z)')

    with open(docs_dir / '0_front_matter_template.md', 'r') as f:
        lines = [
            line.format(version=version, date=date)
            for line in f.readlines()
        ]

    with open(docs_dir / '0_front_matter.md', 'w') as f:
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
