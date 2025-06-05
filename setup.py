# setup.py
#
# Usage:
#    python setup.py build_ext --inplace
#    python setup.py sdist bdist_wheel
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension
from setuptools import setup


MODULE_CIRCUIT: str = 'ck.circuit._circuit_cy'
MODULE_CIRCUIT_TABLE: str = 'ck.pgm_compiler.support.circuit_table._circuit_table_cy'
MODULE_CIRCUIT_ANALYSER: str = 'ck.circuit_compiler.support.circuit_analyser._circuit_analyser_cy'
MODULE_CIRCUIT_COMPILER: str = 'ck.circuit_compiler.cython_vm_compiler._compiler'

CYTHON_MODULES: List[str] = [MODULE_CIRCUIT, MODULE_CIRCUIT_TABLE, MODULE_CIRCUIT_COMPILER]

COMPILER_ARGS: List[str] = []
if sys.platform == 'darwin':
    COMPILER_ARGS += ['-Wno-unreachable-code', '-Wno-unused-function', '-O3']
if sys.platform == 'win32':
    COMPILER_ARGS += ['/O2']

DEFINE_MACROS: List[Tuple[str, str]] = [('NPY_NO_DEPRECATED_API', '1')]

INCLUDE_CK: str = str(Path('src').absolute())
INCLUDE_NP: str = np.get_include()


def pyx(module: str) -> str:
    return 'src/' + module.replace('.', '/') + '.pyx'


CYTHON_EXTENSIONS: List[Extension] = [
    Extension(
        MODULE_CIRCUIT,
        [pyx(MODULE_CIRCUIT)],
        include_dirs=[INCLUDE_NP],
        define_macros=DEFINE_MACROS,
        extra_compile_args=COMPILER_ARGS,
    ),
    Extension(
        MODULE_CIRCUIT_TABLE,
        [pyx(MODULE_CIRCUIT_TABLE)],
        include_dirs=[INCLUDE_NP],
        define_macros=DEFINE_MACROS,
        extra_compile_args=COMPILER_ARGS,
    ),
    Extension(
        MODULE_CIRCUIT_ANALYSER,
        [pyx(MODULE_CIRCUIT_ANALYSER)],
        include_dirs=[INCLUDE_NP],
        define_macros=DEFINE_MACROS,
        extra_compile_args=COMPILER_ARGS,
    ),
    Extension(
        MODULE_CIRCUIT_COMPILER,
        [pyx(MODULE_CIRCUIT_COMPILER)],
        include_dirs=[INCLUDE_NP],
        define_macros=DEFINE_MACROS,
        extra_compile_args=COMPILER_ARGS,
    ),
]

setup(ext_modules=cythonize(CYTHON_EXTENSIONS, include_path=[INCLUDE_CK]))
