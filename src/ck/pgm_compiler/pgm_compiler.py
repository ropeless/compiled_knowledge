from typing import Protocol

from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit


class PGMCompiler(Protocol):
    def __call__(self, pgm: PGM, *, const_parameters: bool = True) -> PGMCircuit:
        """
        A PGM compiler is a function with this signature.

        Args:
            pgm: The PGM to compile.
            const_parameters: If true, the potential function parameters will be circuit
                constants, otherwise they will be circuit variables.

        Returns:
            a PGMCircuit which provides an arithmetic circuit to represent the PGM.
        """
