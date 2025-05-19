import random

from ck import example
from ck.pgm import PGM
from ck.pgm_compiler import factor_elimination
from ck.pgm_circuit.marginals_program import MarginalsProgram
from ck.pgm_circuit import PGMCircuit
from ck.probability.empirical_probability_space import EmpiricalProbabilitySpace

num_of_samples_to_show = 100
num_of_samples_to_calculate = 10_000
rand_seed = None


def main():
    if rand_seed is not None:
        random.seed(rand_seed)

    pgm: PGM = example.Rain()

    pgm_cct: PGMCircuit = factor_elimination.compile_pgm(pgm)
    marginals = MarginalsProgram(pgm_cct)
    sampler = marginals.sample_direct()

    # Show some samples
    for sample in sampler.take(num_of_samples_to_show):
        print(sample)
    print()

    # Show empirical and theoretical marginal distribution for each random variable
    sample_pr = EmpiricalProbabilitySpace(sampler.rvs, sampler.take(num_of_samples_to_calculate))
    for rv in pgm.rvs:
        print(rv, sample_pr.marginal_distribution(rv), marginals.marginal_distribution(rv))
    print()

    print('Done.')


if __name__ == '__main__':
    main()
