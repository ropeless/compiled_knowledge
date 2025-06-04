from ck import example
from ck.pgm import PGM
from ck.pgm_compiler import ace
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.wmc_program import WMCProgram


def main() -> None:

    if not ace.ace_available():
        print("ACE is not available")
        exit(1)

    pgm: PGM = example.Rain()

    # `ace.compile_pgm` will look for an Ace installation in
    # a default location. If Ace is not installed in the default
    # location, then either: (1) pass the location as an argument
    # to `ace.compile_pgm`, or (2) copy the Ace files to
    # the default location using `ace.copy_ace_to_default_location`.
    #
    # Here is an example showing how to copy Ace to the default
    # location from a source directory.
    #
    # ace.copy_ace_to_default_location(r'C:\Research\Ace\ace_v3.0_windows')

    pgm_cct: PGMCircuit = ace.compile_pgm(pgm, print_output=True)

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
