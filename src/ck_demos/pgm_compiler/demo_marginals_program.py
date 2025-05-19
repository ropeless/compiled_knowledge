from ck import example
from ck.pgm import RandomVariable, RVMap
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.marginals_program import MarginalsProgram
from ck.pgm_compiler import factor_elimination, PGMCompiler


COMPILER: PGMCompiler = factor_elimination.compile_pgm_best_jointree


def main() -> None:
    pgm = example.Cancer()

    print(f'Compiling PGM {pgm.name!r} to a Circuit')
    pgm_cct: PGMCircuit = COMPILER(pgm)

    print('Getting Marginals Program')
    marginals = MarginalsProgram(pgm_cct)

    print()
    print('Showing Program results, weighted model count for each instance:')
    for indicators in pgm.instances_as_indicators():
        instance_as_str = pgm.indicator_str(*indicators)
        wmc_value = marginals.wmc(*indicators)
        pgm_value = pgm.value_product_indicators(*indicators)
        print(f'  {instance_as_str:75} {wmc_value:.6f} {pgm_value:.6f}')

    print()
    print('Showing Program results, marginal distribution for each rv:')
    marginals.compute_conditioned()
    for rv in pgm.rvs:
        distribution = marginals.result_for_rv(rv)
        distribution_str = ', '.join(f'{state} = {value:.6f}' for state, value in zip(rv.states, distribution))
        print(f'  {(str(rv) + ":"):12} {distribution_str}')

    print()
    print('Showing Program results, marginal distribution, given cancer = True:')

    cancer: RandomVariable = RVMap(pgm).cancer
    condition = cancer('True')

    marginals.compute_conditioned(condition)
    for rv in pgm.rvs:
        distribution = marginals.result_for_rv(rv)
        distribution_str = ', '.join(f'{state} = {value:.6f}' for state, value in zip(rv.states, distribution))
        print(f'  {(str(rv) + ":"):12} {distribution_str}')

    print()
    print('Done.')


if __name__ == '__main__':
    main()
