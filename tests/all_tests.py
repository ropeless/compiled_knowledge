#!/usr/bin/env python

"""
Unit tests for everything here.
"""
import sys
from pathlib import Path
from typing import List, Iterator
from unittest import TextTestRunner

from tests.helpers.unittest_fixture import make_suit

TEST_VERBOSITY = 2
NUMBER_OF_RUNS = 1


def test_modules() -> List[str]:
    root = Path(__file__).parent
    modules = list(_test_modules_r(root.name, root))
    return modules


def _test_modules_r(prefix: str, folder: Path) -> Iterator[str]:
    for file in folder.iterdir():
        if file.is_dir():
            next_prefix = f'{prefix}.{file.name}'
            for module in _test_modules_r(next_prefix, file):
                yield module
        elif file.is_file() and file.name.endswith('_test.py'):
            yield f'{prefix}.{file.stem}'


def main():
    if NUMBER_OF_RUNS < 0:
        raise ValueError(f'Number of runs cannot be negative: {NUMBER_OF_RUNS}')

    test_modules_to_run = test_modules()
    results = []

    for _ in range(NUMBER_OF_RUNS):
        suite = make_suit(test_modules_to_run)
        result = TextTestRunner(verbosity=TEST_VERBOSITY).run(suite)
        results.append(result)

    if NUMBER_OF_RUNS > 1:
        sys.stderr.flush()
        print()
        print('=' * 70)
        for result in results:
            print(f'Ran {result.testsRun} tests, errors: {len(result.errors)}, failures: {len(result.failures)}')


if __name__ == '__main__':
    main()
