from itertools import repeat, chain
from typing import Sequence, List, Iterator
from unittest import TestCase, main as test_main

from ck.pgm import PGM, RandomVariable, Indicator
from ck.sampling.sampler import Sampler


class SamplerForTesting(Sampler):

    def __init__(self, rvs: Sequence[RandomVariable], samples: List[int], condition: Sequence[Indicator] = ()):
        super().__init__(rvs=rvs, condition=condition)
        self.samples = samples

    def __iter__(self) -> Iterator[int]:
        return chain.from_iterable(repeat(self.samples))


class TestSampler(TestCase):

    def test_rvs(self):
        rv = PGM().new_rv('rv', ['a', 'b', 'c', 'd', 'e', 'f'])
        sampler = SamplerForTesting([rv], [5, 4, 3, 2, 1])
        self.assertEqual(sampler.rvs, (rv,))

    def test_take(self):
        rv = PGM().new_rv('rv', ['a', 'b', 'c', 'd', 'e', 'f'])
        sampler = SamplerForTesting([rv], [5, 4, 3, 2, 1])

        self.assertEqual(
            list(sampler.take(3)),
            [5, 4, 3]
        )
        self.assertEqual(
            list(sampler.take(8)),
            [5, 4, 3, 2, 1, 5, 4, 3],
        )


if __name__ == '__main__':
    test_main()
