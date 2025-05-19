from types import ModuleType
from typing import Tuple

from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_compiler import PGMCompiler


def get_compiler(module: ModuleType, **kwargs) -> Tuple[PGMCompiler]:
    """
    Helper function to create a named PGM compiler.

    Args:
        module: module containing `compile_pgm` function.

    Returns:
        a singleton tuple containing PGMCompiler function.
    """

    def compiler(pgm: PGM, const_parameters: bool = True) -> PGMCircuit:
        """Conforms to the `PGMCompiler` protocol."""
        return module.compile_pgm(pgm, const_parameters=const_parameters, **kwargs)

    return compiler,


def get_compiler_algorithm(module, algorithm: str, **kwargs) -> Tuple[PGMCompiler]:
    """
    Helper function to create a named PGM compiler, with a named algorithm argument.
    """
    return get_compiler(module, algorithm=getattr(module, algorithm, **kwargs))


