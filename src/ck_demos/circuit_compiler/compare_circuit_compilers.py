from typing import Sequence

from ck import example
from ck.circuit_compiler import NamedCircuitCompiler
from ck.pgm import PGM
from ck.pgm_compiler.named_pgm_compilers import NamedPGMCompiler
from ck_demos.utils.compare import compare

PGMS: Sequence[PGM] = [
    example.Rain(),
    # example.Insurance(),
    # example.Pathfinder(),
    # example.Mildew(),
]

PGM_COMPILERS: Sequence[NamedPGMCompiler] = [NamedPGMCompiler.FE_BEST_JOINTREE]

CCT_COMPILERS: Sequence[NamedCircuitCompiler] = list(NamedCircuitCompiler)


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
