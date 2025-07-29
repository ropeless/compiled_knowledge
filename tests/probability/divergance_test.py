from typing import Dict

from ck.pgm import Instance, PGM
from ck.probability.divergence import hi, kl, pseudo_kl, fhi
from ck.probability.pgm_probability_space import PGMProbabilitySpace
from tests.helpers.unittest_fixture import Fixture, test_main
from tests.probability.probability_space_test import DummyProbabilitySpace


class TestDivergence(Fixture):

    def test_kl(self):
        # This test and expected values taken from another implementation of KL

        pgm_a = PGM()
        a_x1 = pgm_a.new_rv('x1', ('no', 'yes'))
        a_x2 = pgm_a.new_rv('x2', ('no', 'yes'))
        a_x3 = pgm_a.new_rv('x3', ('no', 'yes'))
        pgm_a.new_factor(a_x2, a_x1).set_dense().set_flat(0.8, 0.6, 0.2, 0.4)
        pgm_a.new_factor(a_x3, a_x1).set_dense().set_flat(0.9, 0.7, 0.1, 0.3)
        pgm_a.new_factor(a_x1).set_dense().set_flat(0.5, 0.5)
        space_a = PGMProbabilitySpace(pgm_a)

        pgm_b = PGM()
        b_x1 = pgm_b.new_rv('x1', ('no', 'yes'))
        b_x2 = pgm_b.new_rv('x2', ('no', 'yes'))
        b_x3 = pgm_b.new_rv('x3', ('no', 'yes'))
        pgm_b.new_factor(b_x1, b_x2).set_dense().set_flat(4 / 7, 1 / 3, 3 / 7, 2 / 3)
        pgm_b.new_factor(b_x3, b_x2).set_dense().set_flat(57 / 70, 23 / 30, 13 / 70, 7 / 30)
        pgm_b.new_factor(b_x2).set_dense().set_flat(0.7, 0.3)
        space_b = PGMProbabilitySpace(pgm_b)

        self.assertAlmostEqual(kl(space_a, space_b), 0.04468347826465541)
        self.assertAlmostEqual(kl(space_b, space_a), 0.04805616840705247)

    def test_pseudo_kl(self):
        pgm_a = PGM()
        a_x1 = pgm_a.new_rv('x1', ('no', 'yes'))
        a_x2 = pgm_a.new_rv('x2', ('no', 'yes'))
        a_x3 = pgm_a.new_rv('x3', ('no', 'yes'))
        pgm_a.new_factor(a_x2, a_x1).set_dense().set_flat(0.8, 0.6, 0.2, 0.4)
        pgm_a.new_factor(a_x3, a_x1).set_dense().set_flat(0.9, 0.7, 0.1, 0.3)
        pgm_a.new_factor(a_x1).set_dense().set_flat(0.5, 0.5)
        space_a = PGMProbabilitySpace(pgm_a)

        pgm_b = PGM()
        b_x1 = pgm_b.new_rv('x1', ('no', 'yes'))
        b_x2 = pgm_b.new_rv('x2', ('no', 'yes'))
        b_x3 = pgm_b.new_rv('x3', ('no', 'yes'))
        pgm_b.new_factor(b_x1, b_x2).set_dense().set_flat(4 / 7, 1 / 3, 3 / 7, 2 / 3)
        pgm_b.new_factor(b_x3, b_x2).set_dense().set_flat(57 / 70, 23 / 30, 13 / 70, 7 / 30)
        pgm_b.new_factor(b_x2).set_dense().set_flat(0.7, 0.3)
        space_b = PGMProbabilitySpace(pgm_b)

        # TODO `pseudo_kl` is an experimental measure, so we don't bother to test
        #     the returned values. Just checking there are no exceptions raised.
        pseudo_kl(space_a, space_b)
        pseudo_kl(space_b, space_a)

    def test_hi_one_rv_same_pgm(self):
        pgm: PGM = PGM()
        pgm.new_rv('x', 3)
        distribution_1: Dict[Instance, float] = {
            (0,): 2,
            (1,): 3,
            (2,): 5,
        }
        distribution_2: Dict[Instance, float] = {
            (0,): 2,
            (1,): 5,
            (2,): 3,
        }
        space_1 = DummyProbabilitySpace(pgm.rvs, distribution_1.items())
        space_2 = DummyProbabilitySpace(pgm.rvs, distribution_2.items())

        expected = 0.2 + 0.3 + 0.3

        self.assertEqual(hi(space_1, space_2), expected)
        self.assertEqual(hi(space_2, space_1), expected)

    def test_hi_one_rv_difference_pgm(self):
        pgm1: PGM = PGM()
        pgm1.new_rv('x1', 3)
        distribution_1: Dict[Instance, float] = {
            (0,): 2,
            (1,): 3,
            (2,): 5,
        }

        pgm2: PGM = PGM()
        pgm2.new_rv('x2', ('a', 'b', 'c'))
        distribution_2: Dict[Instance, float] = {
            (0,): 2,
            (1,): 5,
            (2,): 3,
        }

        space_1 = DummyProbabilitySpace(pgm1.rvs, distribution_1.items())
        space_2 = DummyProbabilitySpace(pgm2.rvs, distribution_2.items())

        expected = 0.2 + 0.3 + 0.3

        self.assertEqual(hi(space_1, space_2), expected)
        self.assertEqual(hi(space_2, space_1), expected)

    def test_fhi(self):
        # This test and expected values taken from another implementation of FHI

        pgm_a = PGM()
        a_x1 = pgm_a.new_rv('x1', ('no', 'yes'))
        a_x2 = pgm_a.new_rv('x2', ('no', 'yes'))
        a_x3 = pgm_a.new_rv('x3', ('no', 'yes'))
        pgm_a.new_factor(a_x2, a_x1).set_dense().set_flat(0.8, 0.6, 0.2, 0.4)
        pgm_a.new_factor(a_x3, a_x1).set_dense().set_flat(0.9, 0.7, 0.1, 0.3)
        pgm_a.new_factor(a_x1).set_dense().set_flat(0.5, 0.5)
        space_a = PGMProbabilitySpace(pgm_a)

        pgm_b = PGM()
        b_x1 = pgm_b.new_rv('x1', ('no', 'yes'))
        b_x2 = pgm_b.new_rv('x2', ('no', 'yes'))
        b_x3 = pgm_b.new_rv('x3', ('no', 'yes'))
        pgm_b.new_factor(b_x1, b_x2).set_dense().set_flat(4 / 7, 1 / 3, 3 / 7, 2 / 3)
        pgm_b.new_factor(b_x3, b_x2).set_dense().set_flat(57 / 70, 23 / 30, 13 / 70, 7 / 30)
        pgm_b.new_factor(b_x2).set_dense().set_flat(0.7, 0.3)
        space_b = PGMProbabilitySpace(pgm_b)

        self.assertAlmostEqual(fhi(space_a, space_b), 0.9682539682539683)
        self.assertAlmostEqual(fhi(space_b, space_a), 1.0)


if __name__ == '__main__':
    test_main()
