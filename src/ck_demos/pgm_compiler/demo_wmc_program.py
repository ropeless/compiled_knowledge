from ck import example
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import DEFAULT_PGM_COMPILER


def main() -> None:
    pgm = example.Cancer()

    print(f'Compiling PGM {pgm.name!r} to a Circuit')
    pgm_cct: PGMCircuit = DEFAULT_PGM_COMPILER(pgm)

    print('Getting WMC Program')
    wmc = WMCProgram(pgm_cct)

    print()
    print('Showing Program results:')
    for indicators in pgm.instances_as_indicators():
        instance_as_str = pgm.indicator_str(*indicators)
        wmc_value = wmc.wmc(*indicators)
        pgm_value = pgm.value_product_indicators(*indicators)
        print(f'  {instance_as_str:75} {wmc_value:.6f} {pgm_value:.6f}')

    print()
    print('Done.')


if __name__ == '__main__':
    main()
