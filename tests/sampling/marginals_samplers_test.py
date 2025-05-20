from ck.pgm_circuit.marginals_program import MarginalsProgram
from ck.pgm_compiler import factor_elimination
from ck.sampling.sampler import Sampler
from tests.helpers.unittest_fixture import test_main
from tests.helpers.sampling_fixture import SamplingFixture


def mar_sample_direct(pgm, rvs, condition, rand) -> Sampler:
    """
    Conforms to SamplerFunction type.
    """
    mar = MarginalsProgram(factor_elimination.compile_pgm(pgm))
    return mar.sample_direct(
        rvs=rvs,
        condition=condition,
        rand=rand,
    )


def mar_sample_uniform(pgm, rvs, condition, rand) -> Sampler:
    """
    Conforms to SamplerFunction type.
    """
    mar = MarginalsProgram(factor_elimination.compile_pgm(pgm))
    return mar.sample_uniform(
        rvs=rvs,
        condition=condition,
        rand=rand,
    )


def mar_sample_markov(pgm, rvs, condition, rand, chain_pairs, initial_chain_condition) -> Sampler:
    """
    Conforms to MarkovSamplerFunction type.
    """
    mar = MarginalsProgram(factor_elimination.compile_pgm(pgm))
    return mar.sample_direct(
        rvs=rvs,
        condition=condition,
        rand=rand,
        chain_pairs=chain_pairs,
        initial_chain_condition=initial_chain_condition,
    )


class TestMarginalsSampler(SamplingFixture):

    def test_direct_conditioned(self):
        self.check_sampler_conditioned(mar_sample_direct)

    def test_uniform_conditioned(self):
        self.check_sampler_conditioned(mar_sample_uniform)

    def test_direct_unconditioned(self):
        self.check_sampler_unconditioned(mar_sample_direct)

    def test_uniform_unconditioned(self):
        self.check_sampler_unconditioned(mar_sample_uniform)

    def test_markov_chain(self):
        self.check_markov_chain(mar_sample_markov, 0)
        self.check_markov_chain(mar_sample_markov, 1)
        self.check_markov_chain(mar_sample_markov, 2)
        self.check_markov_chain(mar_sample_markov, 3)


if __name__ == '__main__':
    test_main()
