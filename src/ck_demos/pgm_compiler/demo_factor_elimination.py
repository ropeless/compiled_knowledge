from ck import example
from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import factor_elimination
from ck_demos.utils.stop_watch import StopWatch


def main() -> None:
    pgm: PGM = example.Insurance()

    print(f'PGM: {pgm.name}')

    time = StopWatch()
    pgm_cct: PGMCircuit = factor_elimination.compile_pgm_best_jointree(pgm)
    time.stop()
    print(f'time to compile PGM to Circuit:     {time}')

    time.start()
    wmc = WMCProgram(pgm_cct)
    time.stop()
    print(f'time to compile Circuit to Program: {time}')

    time.start()
    for _ in range(1000):
        wmc.compute()
    time.stop()
    print(f'time to execute Program:            {time.seconds() * 1000:,.3f}μs ', end='')

    # print()
    # print(f'Circuit:')
    # pgm_cct.dump()
    # print()
    #
    # print('Showing Program results:')
    # for indicators in pgm.instances_as_indicators():
    #     instance_as_str = pgm.indicator_str(*indicators)
    #     wmc_value = wmc.wmc(*indicators)
    #     pgm_value = pgm.value_product_indicators(*indicators)
    #     print(f'  {instance_as_str:80} {wmc_value:.6f} {pgm_value:.6f}')

    print()
    print('Done.')


if __name__ == '__main__':
    main()
