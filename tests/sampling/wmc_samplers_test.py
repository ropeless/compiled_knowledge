from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import factor_elimination
from ck.sampling.sampler import Sampler
from tests.helpers.unittest_fixture import test_main
from tests.helpers.sampling_fixture import SamplingFixture


def wmc_sample_direct(pgm, rvs, condition, rand) -> Sampler:
    """
    Conforms to SamplerFunction type.
    """
    wmc = WMCProgram(factor_elimination.compile_pgm(pgm))
    return wmc.sample_direct(
        rvs=rvs,
        condition=condition,
        rand=rand,
    )


def wmc_sample_rejection(pgm, rvs, condition, rand) -> Sampler:
    """
    Conforms to SamplerFunction type.
    """
    wmc = WMCProgram(factor_elimination.compile_pgm(pgm))
    return wmc.sample_rejection(
        rvs=rvs,
        condition=condition,
        rand=rand,
    )


def wmc_sample_gibbs(pgm, rvs, condition, rand) -> Sampler:
    """
    Conforms to SamplerFunction type.
    """
    wmc = WMCProgram(factor_elimination.compile_pgm(pgm))
    return wmc.sample_gibbs(
        rvs=rvs,
        condition=condition,
        rand=rand,
    )


def wmc_sample_metropolis(pgm, rvs, condition, rand) -> Sampler:
    """
    Conforms to SamplerFunction type.
    """
    wmc = WMCProgram(factor_elimination.compile_pgm(pgm))
    return wmc.sample_metropolis(
        rvs=rvs,
        condition=condition,
        rand=rand,
    )


def wmc_sample_uniform(pgm, rvs, condition, rand) -> Sampler:
    """
    Conforms to SamplerFunction type.
    """
    wmc = WMCProgram(factor_elimination.compile_pgm(pgm))
    return wmc.sample_uniform(
        rvs=rvs,
        condition=condition,
        rand=rand,
    )


def wmc_sample_markov(pgm, rvs, condition, rand, chain_pairs, initial_chain_condition) -> Sampler:
    """
    Conforms to MarkovSamplerFunction type.
    """
    wmc = WMCProgram(factor_elimination.compile_pgm(pgm))
    return wmc.sample_direct(
        rvs=rvs,
        condition=condition,
        rand=rand,
        chain_pairs=chain_pairs,
        initial_chain_condition=initial_chain_condition,
    )


class TestWMCSampling(SamplingFixture):

    def test_direct_conditioned(self):
        self.check_sampler_conditioned(wmc_sample_direct)

    def test_rejection_conditioned(self):
        self.check_sampler_conditioned(wmc_sample_rejection)

    def test_gibbs_conditioned(self):
        self.check_sampler_conditioned(wmc_sample_gibbs)

    def test_metropolis_conditioned(self):
        self.check_sampler_conditioned(wmc_sample_metropolis)

    def test_uniform_conditioned(self):
        self.check_sampler_conditioned(wmc_sample_uniform)

    def test_direct_unconditioned(self):
        self.check_sampler_unconditioned(wmc_sample_direct)

    def test_rejection_unconditioned(self):
        self.check_sampler_unconditioned(wmc_sample_rejection)

    def test_gibbs_unconditioned(self):
        self.check_sampler_unconditioned(wmc_sample_gibbs)

    def test_metropolis_unconditioned(self):
        self.check_sampler_unconditioned(wmc_sample_metropolis)

    def test_uniform_unconditioned(self):
        self.check_sampler_unconditioned(wmc_sample_uniform)

    def test_markov_chain(self):
        self.check_markov_chain(wmc_sample_markov, 0)
        self.check_markov_chain(wmc_sample_markov, 1)
        self.check_markov_chain(wmc_sample_markov, 2)
        self.check_markov_chain(wmc_sample_markov, 3)


if __name__ == '__main__':
    test_main()
