from typing import Sequence

from ck import example
from ck.circuit_compiler import NamedCircuitCompiler
from ck.pgm import PGM
from ck.pgm_circuit.support.compile_circuit import DEFAULT_CIRCUIT_COMPILER
from ck.pgm_compiler.named_pgm_compilers import NamedPGMCompiler
from ck_demos.utils.compare import compare

# @formatter:off

PGMS: Sequence[PGM] = [
    example.Rain(),
    example.Cancer(),
    example.Earthquake(),
    example.Asia(),
    example.Survey(),
    example.Sachs(),
    example.Child(),
    example.Alarm(),

    # example.Hailfinder(),
    # example.Insurance(),
    # example.Pathfinder(),
    # example.Mildew(),
]

CCT_COMPILERS: Sequence[NamedCircuitCompiler] = [DEFAULT_CIRCUIT_COMPILER]

PGM_COMPILERS: Sequence[NamedPGMCompiler] = [
    named_compiler
    for named_compiler in NamedPGMCompiler
    if named_compiler.name.startswith('FE_') and 'WEIGHTED' not in named_compiler.name
] + [NamedPGMCompiler.ACE]

# @formatter:on


def main() -> None:
    compare(
        pgms=PGMS,
        pgm_compilers=PGM_COMPILERS,
        cct_compilers=CCT_COMPILERS,
    )
    print()
    print('Done.')


if __name__ == '__main__':
    main()
