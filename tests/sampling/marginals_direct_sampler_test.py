from itertools import count
from typing import List

from ck.pgm import PGM, Instance
from ck.pgm_compiler import factor_elimination
from ck.pgm_circuit.marginals_program import MarginalsProgram
from tests.helpers.unittest_fixture import Fixture, test_main


class TestSampler(Fixture):

    def test_markov_chain(self):
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

        mar = MarginalsProgram(factor_elimination.compile_pgm(pgm))

        # Check starting from prev state 0
        sampler = mar.sample_direct(
            rvs=[prev_x, x],
            chain_pairs=[(x, prev_x)],
            initial_chain_condition=prev_x[0],
        )
        expect_samples: List[Instance] = [
            # prev_x, x
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 1),
            (1, 2),
        ]
        for i, sample, expect in zip(count(), sampler, expect_samples):
            self.assertArrayEqual(sample, expect, msg=f'sample number {i}')

        # Check starting from prev state 1
        sampler = mar.sample_direct(
            rvs=[prev_x, x],
            chain_pairs=[(x, prev_x)],
            initial_chain_condition=prev_x[1],
        )
        expect_samples: List[Instance] = [
            # prev_x, x
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 1),
            (1, 2),
            (2, 3),
        ]
        for i, sample, expect in zip(count(), sampler, expect_samples):
            self.assertArrayEqual(sample, expect, msg=f'sample number {i}')

        # Check starting from prev state 2
        sampler = mar.sample_direct(
            rvs=[prev_x, x],
            chain_pairs=[(x, prev_x)],
            initial_chain_condition=prev_x[2],
        )
        expect_samples: List[Instance] = [
            # prev_x, x
            (2, 3),
            (3, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
        ]
        for i, sample, expect in zip(count(), sampler, expect_samples):
            self.assertArrayEqual(sample, expect, msg=f'sample number {i}')

        # Check starting from prev state 3
        sampler = mar.sample_direct(
            rvs=[prev_x, x],
            chain_pairs=[(x, prev_x)],
            initial_chain_condition=prev_x[3],
        )
        expect_samples: List[Instance] = [
            # prev_x, x
            (3, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 1),
        ]
        for i, sample, expect in zip(count(), sampler, expect_samples):
            self.assertArrayEqual(sample, expect, msg=f'sample number {i}')


if __name__ == '__main__':
    test_main()
