from typing import Sequence

from ck.circuit_compiler import NamedCircuitCompiler
from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import NamedPGMCompiler
from ck_demos.utils.stop_watch import StopWatch

CACHE_CIRCUIT: bool = True


def compare(
        pgms: Sequence[PGM],
        pgm_compilers: Sequence[NamedPGMCompiler],
        cct_compilers: Sequence[NamedCircuitCompiler],
) -> None:
    """
    For each combination of the given arguments, construct a PGMCircuit (using a
    PGMCompiler) and then a WMCProgram (using a CircuitCompiler). The resulting
    WMCProgram is executed 1000 times to estimate compute time.

    For each PGM, PGM compiler, and circuit compiler, a line is printed showing:
         PGM,
         PGM compiler name,
         Circuit compiler name,
         number of circuit operations,
         PGMCircuit compile time,
         WMCProgram compile time,
         WMC compute time.

    The print output is formatted using fixed column width.

    Args:
        pgms: a sequence of PGM objects.
        pgm_compilers: a sequence of named PGM compilers.
        cct_compilers: a sequence of named circuit compilers.
    """
    # work out column widths for names.
    max_pgm_name: int = max(len(pgm.name) for pgm in pgms)
    max_pgm_compiler_name: int = max(len(pgm_compiler.name) for pgm_compiler in pgm_compilers)
    max_cct_compiler_name: int = max(len(cct_compiler.name) for cct_compiler in cct_compilers)

    # variables for when CACHE_CIRCUIT is true
    prev_pgm = None
    prev_pgm_compiler = None

    for pgm in pgms:
        pgm_name: str = pgm.name.ljust(max_pgm_name)
        for pgm_compiler in pgm_compilers:
            pgm_compiler_name: str = pgm_compiler.name.ljust(max_pgm_compiler_name)
            for cct_compiler in cct_compilers:
                cct_compiler_name: str = cct_compiler.name.ljust(max_cct_compiler_name)

                print(f'{pgm_name}  ', end='')
                print(f'{pgm_compiler_name}  ', end='')
                print(f'{cct_compiler_name}  ', end='')

                try:
                    time = StopWatch()

                    if CACHE_CIRCUIT and pgm is prev_pgm and pgm_compiler is prev_pgm_compiler:
                        print(f'{"":10} ', end='')
                        print(f'{"":10}  ', end='')
                    else:
                        time.start()
                        pgm_cct: PGMCircuit = pgm_compiler(pgm)
                        time.stop()
                        print(f'{pgm_cct.circuit_top.circuit.number_of_operations:10,} ', end='')
                        print(f'{time.seconds():10.3f}s ', end='')
                        prev_pgm = pgm
                        prev_pgm_compiler = pgm_compiler

                    time.start()
                    wmc = WMCProgram(pgm_cct, compiler=cct_compiler.compiler)
                    time.stop()
                    print(f'{time.seconds():10.3f}s ', end='')

                    time.start()
                    for _ in range(1000):
                        wmc.compute()
                    time.stop()
                    print(f'{time.seconds() * 1000:10.3f}μs ', end='')
                except Exception as err:
                    print(repr(err), end='')

                print()
        print()
