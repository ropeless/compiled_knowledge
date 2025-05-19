from ck.circuit_compiler import llvm_compiler
from ck import example
from ck.pgm_compiler import factor_elimination
from ck.pgm_circuit import PGMCircuit
from ck.program.program_buffer import ProgramBuffer


PGM_COMPILER = factor_elimination.compile_pgm_best_jointree
CCT_COMPILER = llvm_compiler.compile_circuit


def main() -> None:
    pgm = example.Rain()

    print(f'Compiling PGM {pgm.name!r} to a Circuit')
    pgm_cct: PGMCircuit = PGM_COMPILER(pgm)

    print('Compiling Circuit to a Program')
    prog = ProgramBuffer(CCT_COMPILER(pgm_cct.circuit_top))
    slot_map = pgm_cct.slot_map

    print('Showing Program results')
    for indicators in pgm.instances_as_indicators():
        prog[:] = 0
        for ind in indicators:
            prog[slot_map[ind]] = 1

        instance_as_str = pgm.indicator_str(*indicators)
        program_value = prog.compute().item()
        pgm_value = pgm.value_product_indicators(*indicators)
        print(f'  {instance_as_str:75} {program_value:.6f} {pgm_value:.6f}')

    print()
    print('Done.')


if __name__ == '__main__':
    main()
