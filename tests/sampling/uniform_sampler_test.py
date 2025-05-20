from ck.sampling.sampler import Sampler
from ck.sampling.uniform_sampler import UniformSampler
from tests.helpers.unittest_fixture import test_main
from tests.helpers.sampling_fixture import SamplingFixture


def uniform_sampler(_, rvs, condition, rand) -> Sampler:
    """
    Conforms to SamplerFunction type.
    """
    return UniformSampler(
        rvs=rvs,
        condition=condition,
        rand=rand,
    )


class TestWMCSampling(SamplingFixture):

    def test_uniform_conditioned(self):
        self.check_sampler_conditioned(uniform_sampler)

    def test_uniform_unconditioned(self):
        self.check_sampler_unconditioned(uniform_sampler)


if __name__ == '__main__':
    test_main()
