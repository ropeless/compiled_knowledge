import random
from itertools import count
from typing import List, TypeAlias, Callable, Sequence, Tuple

from ck.pgm import PGM, Instance, RandomVariable
from ck.probability.probability_space import Condition
from ck.sampling.sampler import Sampler
from ck.utils.random_extras import Random
from tests.helpers.unittest_fixture import Fixture

SamplerFunction: TypeAlias = Callable[
    [
        PGM,  # pgm
        Sequence[RandomVariable],  # sample rvs
        Condition,  # condition
        Random,  # rand
    ],
    Sampler
]

MarkovSamplerFunction: TypeAlias = Callable[
    [
        PGM,  # pgm
        Sequence[RandomVariable],  # sample rvs
        Condition,  # condition
        Random,  # rand
        Sequence[Tuple[RandomVariable, RandomVariable]],  # chain_pairs
        Condition,  # initial_chain_condition
    ],
    Sampler
]


class SamplingFixture(Fixture):

    def check_sampler_conditioned(self, sampler_function: SamplerFunction) -> None:
        rv_states = [2, 3, 5, 6, 4]  # number of states for each random variable
        num_of_samples = 100  # must be at least 2

        seed = 123123178243
        random.seed(seed)
        # noinspection PyTypeChecker
        rand: Random = random

        # create a random PGM
        pgm = PGM()
        rvs = [pgm.new_rv(f'x_{i}', num_states) for i, num_states in enumerate(rv_states)]
        pgm.new_factor(*rvs).set_dense().set_stream(random.random)

        # get sampler
        condition_1 = (rvs[1][0],)
        condition_2 = (rvs[2][1], rvs[2][2])
        condition = condition_1 + condition_2
        sampler = sampler_function(pgm, rvs, condition, rand)

        self.assertEqual(sampler.rvs, tuple(rvs))

        for _, states in zip(range(num_of_samples), sampler):
            self.assertEqual(len(states), len(rvs))
            for rv, state in zip(rvs, states):
                self.assertIn(state, rv.state_range())

            # condition 1
            self.assertEqual(states[1], 0)

            # condition 2
            self.assertIn(states[2], (1, 2))

    def check_sampler_unconditioned(self, sampler_function: SamplerFunction) -> None:
        rv_states = [2, 3, 5, 6, 4]  # number of states for each random variable
        num_of_samples = 100  # must be at least 2

        seed = 123123178243
        random.seed(seed)
        # noinspection PyTypeChecker
        rand: Random = random

        # create a random PGM
        pgm = PGM()
        rvs = [pgm.new_rv(f'x_{i}', num_states) for i, num_states in enumerate(rv_states)]
        pgm.new_factor(*rvs).set_dense().set_stream(random.random)

        # get sampler
        sampler = sampler_function(pgm, rvs, (), rand)

        self.assertEqual(sampler.rvs, tuple(rvs))

        for _, states in zip(range(num_of_samples), sampler):
            self.assertEqual(len(states), len(rvs))
            for rv, state in zip(rvs, states):
                self.assertIn(state, rv.state_range())

    def check_markov_chain(self, sampler_function: MarkovSamplerFunction, start_state: int) -> None:
        """
        Check that a deterministic Markov chain is consistent with expected states.

        Creates a PGM with Deterministic state transition from 'prev_x' to 'x':
            0 -> 1
            1 -> 2
            2 -> 3
            3 -> 0

        The Markov chain is sampled, starting from 'start_state' (0, 1, 2, 3)
        """
        assert 0 <= start_state < 4, 'must have 0 <= start_state < 4'

        pgm = PGM()
        x = pgm.new_rv('x', 4)
        prev_x = pgm.new_rv('prev_x', 4)

        # Uniform random initial state for `prev_x`
        pgm.new_factor(prev_x).set_dense().set_uniform()

        # Deterministic state transition from 'prev_x' to 'x':
        # 0 -> 1
        # 1 -> 2
        # 2 -> 3
        # 3 -> 0
        f = pgm.new_factor(x, prev_x).set_cpt()
        f.set_cpd(0, [0, 1, 0, 0])
        f.set_cpd(1, [0, 0, 1, 0])
        f.set_cpd(2, [0, 0, 0, 1])
        f.set_cpd(3, [1, 0, 0, 0])

        seed = 123123178243
        random.seed(seed)
        # noinspection PyTypeChecker
        rand: Random = random

        # Check starting from prev start_state
        rvs = [prev_x, x]
        chain_pairs = [(x, prev_x)]
        condition = ()
        initial_chain_condition = prev_x[start_state]
        sampler = sampler_function(pgm, rvs, condition, rand, chain_pairs, initial_chain_condition)

        expect_cycle: List[Instance] = [
            # prev_x, x
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
        ]

        expect_samples: List[Instance] = (expect_cycle * 4)[start_state:]

        for i, sample, expect in zip(count(), sampler, expect_samples):
            self.assertArrayEqual(sample, expect, msg=f'sample number {i}')
