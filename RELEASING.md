Release Process
===============

Prepare for the release
-----------------------

1. Check out the "main" branch.
2. Merge any new features and fixes for the release.
3. Run `python setup.py build_ext --inplace` to compile Cython modules (must do before testing).
4. Run `tests.all_tests.py` for unit testing, confirming all unit tests pass.
5. Run `ck_demos.all_demos.py` for smoke testing, confirming no errors are reported.
6. Edit `pyproject.toml` to update the project version number (must do before building documentation).
7. Run `python build_docs.py` and confirm the documentation builds okay.
8. View the documentation build to confirm it: [`docs/_build/html/index.html`](docs/_build/html/index.html).

Only proceed if the "main" branch is ready for release.

Perform the release
-------------------

1. Commit and push the "main" branch.
   This will automatically release the documentation.
2. Ensure you are on an up-to-date checkout of the "main" branch.
3. Delete any existing project `dist` directory.
4. Build the package using: `python setup.py sdist bdist_wheel`.
5. Upload the package to PyPI using: `python -m twine upload dist/*`.


Post-release checks
-------------------

1. Check the online version of the documentation:  https://compiled-knowledge.readthedocs.io/.
2. Check the PyPi release history: https://pypi.org/project/compiled-knowledge/#history.
3. Open a CK client test project. Update the dependencies (e.g., `poetry update`).
   Ensure CK upgraded and that the test project works as expected.

Actions for a broken release
============================

If the post-release checks fail:

1. Delete the broken PyPi release: https://pypi.org/project/compiled-knowledge/#history.
2. Create a "fix/..." branch for a fix.
3. When a fix is found, perform the release process above.
