import random

from ck import example
from ck.pgm import PGM
from ck.pgm_compiler import factor_elimination
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.probability.empirical_probability_space import EmpiricalProbabilitySpace

num_of_samples = 100
seed = None


def main():
    if seed is not None:
        random.seed(seed)

    pgm: PGM = example.Rain()
    pgm_cct: PGMCircuit = factor_elimination.compile_pgm(pgm)
    wmc = WMCProgram(pgm_cct)
    sampler = wmc.sample_uniform()

    # Show some samples
    for sample in sampler.take(num_of_samples):
        print(sample)
    print()

    # Show empirical marginal distribution for each random variable
    pr = EmpiricalProbabilitySpace(sampler.rvs, sampler.take(100_000))
    for rv in pgm.rvs:
        print(rv, pr.marginal_distribution(rv))
    print()

    print('Done.')


if __name__ == '__main__':
    main()
