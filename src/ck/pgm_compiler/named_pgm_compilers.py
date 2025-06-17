from enum import Enum

from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_compiler import variable_elimination, factor_elimination, recursive_conditioning, ace
from .pgm_compiler import PGMCompiler
from .support.named_compiler_maker import get_compiler_algorithm as _get_compiler_algorithm, \
    get_compiler as _get_compiler


class NamedPGMCompiler(Enum):
    """
    A standard collection of named compiler functions.

    The `value` of each enum member is tuple containing a compiler function (PGM -> PGMCircuit).
    Wrapping in a tuple is needed otherwise Python erases the type of the member, which can cause problems.
    Each member itself is callable, confirming to the PGMCompiler protocol, delegating to the compiler function.
    """

    VE_MIN_DEGREE = _get_compiler_algorithm(variable_elimination, 'MIN_DEGREE')
    VE_MIN_DEGREE_THEN_FILL = _get_compiler_algorithm(variable_elimination, 'MIN_DEGREE_THEN_FILL')
    VE_MIN_FILL = _get_compiler_algorithm(variable_elimination, 'MIN_FILL')
    VE_MIN_FILL_THEN_DEGREE = _get_compiler_algorithm(variable_elimination, 'MIN_FILL_THEN_DEGREE')
    VE_MIN_WEIGHTED_DEGREE = _get_compiler_algorithm(variable_elimination, 'MIN_WEIGHTED_DEGREE')
    VE_MIN_WEIGHTED_FILL = _get_compiler_algorithm(variable_elimination, 'MIN_WEIGHTED_FILL')
    VE_MIN_TRADITIONAL_WEIGHTED_FILL = _get_compiler_algorithm(variable_elimination, 'MIN_TRADITIONAL_WEIGHTED_FILL')

    FE_MIN_DEGREE = _get_compiler_algorithm(factor_elimination, 'MIN_DEGREE')
    FE_MIN_DEGREE_THEN_FILL = _get_compiler_algorithm(factor_elimination, 'MIN_DEGREE_THEN_FILL')
    FE_MIN_FILL = _get_compiler_algorithm(factor_elimination, 'MIN_FILL')
    FE_MIN_FILL_THEN_DEGREE = _get_compiler_algorithm(factor_elimination, 'MIN_FILL_THEN_DEGREE')
    FE_MIN_WEIGHTED_DEGREE = _get_compiler_algorithm(factor_elimination, 'MIN_WEIGHTED_DEGREE')
    FE_MIN_WEIGHTED_FILL = _get_compiler_algorithm(factor_elimination, 'MIN_WEIGHTED_FILL')
    FE_MIN_TRADITIONAL_WEIGHTED_FILL = _get_compiler_algorithm(factor_elimination, 'MIN_TRADITIONAL_WEIGHTED_FILL')
    FE_BEST_JOINTREE = factor_elimination.compile_pgm_best_jointree,

    RC_MIN_DEGREE = _get_compiler_algorithm(recursive_conditioning, 'MIN_DEGREE')
    RC_MIN_DEGREE_THEN_FILL = _get_compiler_algorithm(recursive_conditioning, 'MIN_DEGREE_THEN_FILL')
    RC_MIN_FILL = _get_compiler_algorithm(recursive_conditioning, 'MIN_FILL')
    RC_MIN_FILL_THEN_DEGREE = _get_compiler_algorithm(recursive_conditioning, 'MIN_FILL_THEN_DEGREE')
    RC_MIN_WEIGHTED_DEGREE = _get_compiler_algorithm(recursive_conditioning, 'MIN_WEIGHTED_DEGREE')
    RC_MIN_WEIGHTED_FILL = _get_compiler_algorithm(recursive_conditioning, 'MIN_WEIGHTED_FILL')
    RC_MIN_TRADITIONAL_WEIGHTED_FILL = _get_compiler_algorithm(recursive_conditioning, 'MIN_TRADITIONAL_WEIGHTED_FILL')

    ACE = _get_compiler(ace)

    def __call__(self, pgm: PGM, const_parameters: bool = True) -> PGMCircuit:
        """
        Each member of the enum is a PGMCompiler function.

        This implements the `PGMCompiler` protocol.
        """
        return self.compiler(pgm, const_parameters=const_parameters)

    @property
    def compiler(self) -> PGMCompiler:
        return self.value[0]


DEFAULT_PGM_COMPILER: NamedPGMCompiler = NamedPGMCompiler.FE_BEST_JOINTREE
