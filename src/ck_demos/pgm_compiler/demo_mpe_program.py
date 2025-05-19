from ck import example
from ck.pgm import RVMap, RandomVariable
from ck.pgm_circuit.mpe_program import MPEProgram, MPEResult
from ck.pgm_compiler import factor_elimination,  PGMCompiler
from ck.pgm_circuit import PGMCircuit


COMPILER: PGMCompiler = factor_elimination.compile_pgm_best_jointree


def main() -> None:
    pgm = example.Cancer()

    rv_map = RVMap(pgm)
    pollution: RandomVariable = rv_map.pollution
    smoker: RandomVariable = rv_map.smoker
    cancer: RandomVariable = rv_map.cancer
    xray: RandomVariable = rv_map.xray
    dyspnoea: RandomVariable = rv_map.dyspnoea

    print(f'Compiling PGM {pgm.name!r} to a Circuit')
    pgm_cct: PGMCircuit = COMPILER(pgm)

    print('Getting MPE Program')
    mpe = MPEProgram(pgm_cct)

    print()
    print('Showing Program results:')

    conditions = [
        [],
        [smoker('True')],
        [cancer('True')],
        [pollution('low')],
        [xray('positive')],
        [pollution('high'), dyspnoea('False')],
    ]

    z: float = pgm.value_product_indicators()

    for condition in conditions:
        result: MPEResult = mpe.mpe(*condition)

        condition_str = pgm.condition_str(*condition)
        result_str = pgm.instance_str(result.mpe)
        pr = result.wmc / z

        print(f'MPE[{condition_str}] = {result_str} with pr = {pr}')

    print()
    print('Done.')


if __name__ == '__main__':
    main()
