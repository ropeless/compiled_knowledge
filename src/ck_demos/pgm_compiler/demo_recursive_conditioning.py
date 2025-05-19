from ck import example
from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import recursive_conditioning


def main() -> None:
    pgm: PGM = example.Rain()

    pgm_cct: PGMCircuit = recursive_conditioning.compile_pgm(pgm)

    print(f'PGM: {pgm.name}')
    print()
    print(f'Circuit:')
    pgm_cct.dump()
    print()

    wmc = WMCProgram(pgm_cct)

    print('Showing Program results:')
    for indicators in pgm.instances_as_indicators():
        instance_as_str = pgm.indicator_str(*indicators)
        wmc_value = wmc.wmc(*indicators)
        pgm_value = pgm.value_product_indicators(*indicators)
        print(f'  {instance_as_str:80} {wmc_value:.6f} {pgm_value:.6f}')

    print()
    print('Done.')


if __name__ == '__main__':
    main()
