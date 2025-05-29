import gc
from typing import Sequence

from ck.circuit_compiler import NamedCircuitCompiler
from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import NamedPGMCompiler
from ck_demos.utils.stop_watch import StopWatch


def compare(
        pgms: Sequence[PGM],
        pgm_compilers: Sequence[NamedPGMCompiler],
        cct_compilers: Sequence[NamedCircuitCompiler],
        *,
        cache_circuits: bool = True,
        break_between_pgms: bool = True,
        comma_numbers: bool = True,
        print_header: bool = True,
        sep: str = '  ',
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
        cache_circuits: if true, then circuits are reused across different circuit compilers.
        break_between_pgms: if true, print a blank line  between different workload PGMs.
        comma_numbers: if true, commas are used in large numbers.
        print_header: if true, a header line is printed.
        sep: column separator.
    """
    # Work out column widths for names.
    col_pgm_name: int = max(3, max(len(pgm.name) for pgm in pgms))
    col_pgm_compiler_name: int = max(12, max(len(pgm_compiler.name) for pgm_compiler in pgm_compilers))
    col_cct_compiler_name: int = max(12, max(len(cct_compiler.name) for cct_compiler in cct_compilers))
    col_cct_ops: int = 10
    col_pgm_compile_time: int = 16
    col_cct_compile_time: int = 16
    col_execute_time: int = 10

    # Print formatting
    comma: str = ',' if comma_numbers else ''

    if print_header:
        print('PGM'.ljust(col_pgm_name), end=sep)
        print('PGM-compiler'.ljust(col_pgm_compiler_name), end=sep)
        print('CCT-compiler'.ljust(col_cct_compiler_name), end=sep)
        print('CCT-ops'.rjust(col_cct_ops), end=sep)
        print('PGM-compile-time'.rjust(col_pgm_compile_time), end=sep)
        print('CCT-compile-time'.rjust(col_cct_compile_time), end=sep)
        print('Run-time'.rjust(col_execute_time))

    # Variables for when cache_circuits is true
    prev_pgm = None
    prev_pgm_compiler = None

    for pgm in pgms:
        pgm_name: str = pgm.name.ljust(col_pgm_name)
        for pgm_compiler in pgm_compilers:
            pgm_compiler_name: str = pgm_compiler.name.ljust(col_pgm_compiler_name)
            for cct_compiler in cct_compilers:
                cct_compiler_name: str = cct_compiler.name.ljust(col_cct_compiler_name)

                print(pgm_name, end=sep)
                print(pgm_compiler_name, end=sep)
                print(cct_compiler_name, end=sep)

                try:
                    time = StopWatch()

                    if cache_circuits and pgm is prev_pgm and pgm_compiler is prev_pgm_compiler:
                        print(f'{"":{col_cct_ops}}', end=sep)
                        print(f'{"":{col_pgm_compile_time}}', end=sep)
                    else:
                        gc.collect()
                        time.start()
                        pgm_cct: PGMCircuit = pgm_compiler(pgm)
                        time.stop()
                        num_ops: int = pgm_cct.circuit_top.circuit.number_of_operations
                        print(f'{num_ops:{col_cct_ops}{comma}}', end=sep)
                        print(f'{time.seconds():{col_pgm_compile_time}{comma}.3f}', end=sep)
                        prev_pgm = pgm
                        prev_pgm_compiler = pgm_compiler

                    gc.collect()
                    time.start()
                    # `pgm_cct` will always be set but the IDE can't work that out.
                    # noinspection PyUnboundLocalVariable
                    wmc = WMCProgram(pgm_cct, compiler=cct_compiler.compiler)
                    time.stop()
                    print(f'{time.seconds():{col_cct_compile_time}{comma}.3f}', end=sep)

                    gc.collect()
                    time.start()
                    for _ in range(1000):
                        wmc.compute()
                    time.stop()
                    print(f'{time.seconds() * 1000:{col_execute_time}{comma}.3f}', end='')
                except Exception as err:
                    print(repr(err), end='')
                print()
        if break_between_pgms:
            print()
