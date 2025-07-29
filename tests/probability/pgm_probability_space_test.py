import math
from unittest import main as test_main

from ck.pgm import PGM
from ck.probability.pgm_probability_space import PGMProbabilitySpace
from tests.helpers.unittest_fixture import Fixture


class TestPGMProbabilitySpace(Fixture):
    """
    These unit tests not only cover the functionality of PGMProbabilitySpace,
    but they also indirectly cover ProbabilitySpace, as the behaviour of
    PGM is well-defined and tested elsewhere.
    """

    def test_no_rv(self):
        pgm = PGM()
        pr = PGMProbabilitySpace(pgm)

        self.assertArrayEqual(pr.rvs, [])
        self.assertEqual(pr.z, 1)
        self.assertEqual(pr.wmc(), 1)

        self.assertEqual(pr.probability(), 1)

    def test_one_rv(self):
        pgm = PGM()
        x = pgm.new_rv('x', ['a', 'b', 'c'])
        pgm.new_factor(x).set_dense().set_flat(2, 3, 7)
        pr = PGMProbabilitySpace(pgm)

        self.assertArrayEqual(pr.rvs, [x])
        self.assertEqual(pr.z, 12)

        self.assertEqual(pr.wmc(), 12)
        self.assertEqual(pr.wmc(x[0]), 2)
        self.assertEqual(pr.wmc(x[1]), 3)
        self.assertEqual(pr.wmc(x[2]), 7)
        self.assertEqual(pr.wmc(x[0], x[1]), 5)
        self.assertEqual(pr.wmc(x[0], x[2]), 9)
        self.assertEqual(pr.wmc(x[1], x[2]), 10)
        self.assertEqual(pr.wmc(x[0], x[1], x[2]), 12)

        self.assertEqual(pr.probability(), 1)
        self.assertEqual(pr.probability(x[0]), 2 / 12)
        self.assertEqual(pr.probability(x[1]), 3 / 12)
        self.assertEqual(pr.probability(x[2]), 7 / 12)
        self.assertEqual(pr.probability(x[0], x[1]), 5 / 12)
        self.assertEqual(pr.probability(x[0], x[2]), 9 / 12)
        self.assertEqual(pr.probability(x[1], x[2]), 10 / 12)
        self.assertEqual(pr.probability(x[0], x[1], x[2]), 1)

        self.assertEqual(pr.probability(condition=x[0]), 1)
        self.assertEqual(pr.probability(x[0], condition=x[0]), 1)
        self.assertEqual(pr.probability(x[1], condition=x[0]), 0)
        self.assertEqual(pr.probability(x[2], condition=x[0]), 0)
        self.assertEqual(pr.probability(x[0], condition=(x[0], x[1])), 2 / (2 + 3))
        self.assertEqual(pr.probability(x[1], condition=(x[0], x[1])), 3 / (2 + 3))
        self.assertEqual(pr.probability(x[2], condition=(x[0], x[1])), 0)
        self.assertEqual(pr.probability(x[0], x[1], condition=x[1]), 1)
        self.assertEqual(pr.probability(x[0], x[2], condition=x[1]), 0)
        self.assertEqual(pr.probability(x[0], x[1], condition=(x[0], x[1])), 1)
        self.assertEqual(pr.probability(x[0], x[2], condition=(x[0], x[1])), 2 / (2 + 3))

    def test_two_rvs(self):
        pgm = PGM()
        x = pgm.new_rv('x', ['a', 'b'])
        y = pgm.new_rv('y', ['c', 'd'])
        pgm.new_factor(x).set_dense().set_flat(2, 3)
        pgm.new_factor(y).set_dense().set_flat(7, 11)
        pr = PGMProbabilitySpace(pgm)

        expect_z = 2 * 7 + 2 * 11 + 3 * 7 + 3 * 11
        sum_x = 2 + 3
        sum_y = 7 + 11

        self.assertEqual(pr.z, expect_z)

        self.assertEqual(pr.wmc(), expect_z)
        self.assertEqual(pr.wmc(x[0]), 2 * sum_y)
        self.assertEqual(pr.wmc(x[1]), 3 * sum_y)
        self.assertEqual(pr.wmc(y[0]), 7 * sum_x)
        self.assertEqual(pr.wmc(y[1]), 11 * sum_x)

        self.assertEqual(pr.wmc(x[0], x[1]), expect_z)
        self.assertEqual(pr.wmc(y[0], y[1]), expect_z)
        self.assertEqual(pr.wmc(x[0], y[0]), 2 * 7)
        self.assertEqual(pr.wmc(x[0], y[1]), 2 * 11)
        self.assertEqual(pr.wmc(x[1], y[0]), 3 * 7)
        self.assertEqual(pr.wmc(x[1], y[1]), 3 * 11)

        self.assertEqual(pr.wmc(x[0], y[0], y[1]), 2 * sum_y)
        self.assertEqual(pr.wmc(x[1], y[0], y[1]), 3 * sum_y)
        self.assertEqual(pr.wmc(y[0], x[0], x[1]), 7 * sum_x)
        self.assertEqual(pr.wmc(y[1], x[0], x[1]), 11 * sum_x)

        self.assertEqual(pr.wmc(x[0], x[1], y[0], y[1]), expect_z)

        self.assertEqual(pr.probability(), 1)
        self.assertEqual(pr.probability(x[0]), 2 * sum_y / expect_z)
        self.assertEqual(pr.probability(x[1]), 3 * sum_y / expect_z)
        self.assertEqual(pr.probability(y[0]), 7 * sum_x / expect_z)
        self.assertEqual(pr.probability(y[1]), 11 * sum_x / expect_z)

        self.assertEqual(pr.probability(x[0], x[1]), 1)
        self.assertEqual(pr.probability(y[0], y[1]), 1)
        self.assertEqual(pr.probability(x[0], y[0]), 2 * 7 / expect_z)
        self.assertEqual(pr.probability(x[0], y[1]), 2 * 11 / expect_z)
        self.assertEqual(pr.probability(x[1], y[0]), 3 * 7 / expect_z)
        self.assertEqual(pr.probability(x[1], y[1]), 3 * 11 / expect_z)

        self.assertEqual(pr.probability(x[0], y[0], y[1]), 2 * sum_y / expect_z)
        self.assertEqual(pr.probability(x[1], y[0], y[1]), 3 * sum_y / expect_z)
        self.assertEqual(pr.probability(y[0], x[0], x[1]), 7 * sum_x / expect_z)
        self.assertEqual(pr.probability(y[1], x[0], x[1]), 11 * sum_x / expect_z)

        self.assertEqual(pr.probability(x[0], x[1], y[0], y[1]), 1)

        self.assertEqual(pr.probability(condition=x[0]), 1)
        self.assertEqual(pr.probability(x[0], condition=x[0]), 1)
        self.assertEqual(pr.probability(x[1], condition=x[0]), 0)
        self.assertEqual(pr.probability(y[0], condition=x[0]), 7 / sum_y)
        self.assertEqual(pr.probability(y[1], condition=x[0]), 11 / sum_y)

        self.assertEqual(pr.probability(condition=(x[0], y[1])), 1)
        self.assertEqual(pr.probability(x[0], condition=(x[0], y[1])), 1)
        self.assertEqual(pr.probability(x[1], condition=(x[0], y[1])), 0)
        self.assertEqual(pr.probability(y[0], condition=(x[0], y[1])), 0)
        self.assertEqual(pr.probability(y[1], condition=(x[0], y[1])), 1)

    def test_marginals_simple(self):
        pgm = PGM()
        x = pgm.new_rv('x', ['a', 'b', 'c'])
        pgm.new_factor(x).set_dense().set_flat(0.2, 0.5, 0.3)
        pr = PGMProbabilitySpace(pgm)

        marginals = pr.marginal_distribution(x)
        self.assertArrayEqual(marginals, [0.2, 0.5, 0.3])

    def test_marginals(self):
        pgm = PGM()
        x = pgm.new_rv('x', ['a', 'b'])
        y = pgm.new_rv('y', ['c', 'd'])
        pgm.new_factor(x).set_dense().set_flat(2, 3)
        pgm.new_factor(y).set_dense().set_flat(7, 11)
        pr = PGMProbabilitySpace(pgm)

        marginals = pr.marginal_distribution(x)
        self.assertArrayEqual(marginals, [2 / 5, 3 / 5])

        marginals = pr.marginal_distribution(y)
        self.assertArrayEqual(marginals, [7 / 18, 11 / 18])

        marginals = pr.marginal_distribution(x, y)
        expect_z = 2 * 7 + 2 * 11 + 3 * 7 + 3 * 11
        expect = [
            2 * 7 / expect_z,
            2 * 11 / expect_z,
            3 * 7 / expect_z,
            3 * 11 / expect_z,
        ]
        self.assertArrayEqual(marginals, expect)

    def test_correlation(self):
        pgm = PGM()
        x = pgm.new_rv('x', ['a', 'b'])
        y = pgm.new_rv('y', ['c', 'd'])
        pgm.new_factor(x).set_dense().set_flat(0.2, 0.8)
        pgm.new_factor(y).set_dense().set_flat(0.5, 0.5)
        pr = PGMProbabilitySpace(pgm)

        p1 = 0.2
        p2 = 0.5
        p12 = p1 * p2
        r = (p12 - p1 * p2) / math.sqrt(p1 * (1.0 - p1) * p2 * (1.0 - p2))

        self.assertAlmostEqual(pr.correlation(x[0], y[0]), r)

    def test_correlation_with_zero(self):
        pgm = PGM()
        x = pgm.new_rv('x', ['a', 'b'])
        y = pgm.new_rv('y', ['c', 'd'])
        pgm.new_factor(x).set_dense().set_flat(0.2, 0.8)
        pgm.new_factor(y).set_dense().set_flat(0.0, 1.0)
        pr = PGMProbabilitySpace(pgm)

        self.assertEqual(pr.correlation(x[0], y[0]), 0)
        self.assertEqual(pr.correlation(x[0], y[1]), 0)

    def test_entropy(self):
        pgm = PGM()
        x = pgm.new_rv('x', ['a', 'b'])
        y = pgm.new_rv('y', ['c', 'd'])
        pgm.new_factor(x).set_dense().set_flat(0.2, 0.8)
        pgm.new_factor(y).set_dense().set_flat(0.5, 0.5)
        pr = PGMProbabilitySpace(pgm)

        self.assertAlmostEqual(pr.entropy(x), - 0.2 * math.log2(0.2) - 0.8 * math.log2(0.8))
        self.assertAlmostEqual(pr.entropy(y), - 0.5 * math.log2(0.5) - 0.5 * math.log2(0.5))

    def test_entropy_with_zero(self):
        pgm = PGM()
        x = pgm.new_rv('x', ['a', 'b'])
        y = pgm.new_rv('y', ['c', 'd'])
        pgm.new_factor(x).set_dense().set_flat(0.2, 0.8)
        pgm.new_factor(y).set_dense().set_flat(0.0, 1.0)
        pr = PGMProbabilitySpace(pgm)

        self.assertEqual(pr.entropy(y), 0)

    def test_zero_weight_world(self):
        pgm = PGM()
        x = pgm.new_rv('x', ['a', 'b'])
        y = pgm.new_rv('y', ['c', 'd'])
        f = pgm.new_factor(x, y).set_dense()
        f[0, 0] = 2
        f[0, 1] = 3
        f[1, 0] = 0
        f[1, 1] = 5
        pr = PGMProbabilitySpace(pgm)

        self.assertEqual(pr.z, 10)
        self.assertEqual(pr.wmc(), 10)

        self.assertEqual(pr.probability(), 1)

        self.assertEqual(pr.probability(x[0], y[0]), 0.2)
        self.assertEqual(pr.probability(x[0], y[1]), 0.3)
        self.assertEqual(pr.probability(x[1], y[0]), 0.0)
        self.assertEqual(pr.probability(x[1], y[1]), 0.5)

        self.assertEqual(pr.probability(condition=x[1]), 1)
        self.assertEqual(pr.probability(x[0], condition=x[1]), 0)
        self.assertEqual(pr.probability(x[1], condition=x[1]), 1)
        self.assertEqual(pr.probability(y[0], condition=x[1]), 0)
        self.assertEqual(pr.probability(y[1], condition=x[1]), 1)

        self.assertTrue(math.isnan(pr.probability(x[0], condition=(x[1], y[0]))))
        self.assertTrue(math.isnan(pr.probability(x[1], condition=(x[1], y[0]))))
        self.assertTrue(math.isnan(pr.probability(y[0], condition=(x[1], y[0]))))
        self.assertTrue(math.isnan(pr.probability(y[1], condition=(x[1], y[0]))))

    def test_zero_potential_function_one_rv(self):
        pgm = PGM()
        x = pgm.new_rv('x', ['a', 'b', 'c'])
        pgm.new_factor(x)
        pr = PGMProbabilitySpace(pgm)

        self.assertEqual(pr.z, 0)
        self.assertEqual(pr.wmc(), 0)

        self.assertTrue(math.isnan(pr.probability()))
        self.assertTrue(math.isnan(pr.probability(condition=x[0])))
        self.assertTrue(math.isnan(pr.probability(x[0], condition=x[0])))

        marginals = pr.marginal_distribution(x)
        self.assertEqual(len(marginals), 3)
        self.assertTrue(math.isnan(marginals[0]))
        self.assertTrue(math.isnan(marginals[1]))
        self.assertTrue(math.isnan(marginals[2]))

    def test_zero_potential_function_two_rvs(self):
        pgm = PGM()
        x = pgm.new_rv('x', ['a', 'b', 'c'])
        y = pgm.new_rv('y', ['d', 'e'])
        pgm.new_factor(x)
        pgm.new_factor(y)
        pr = PGMProbabilitySpace(pgm)

        self.assertEqual(pr.z, 0)
        self.assertEqual(pr.wmc(), 0)

        self.assertTrue(math.isnan(pr.probability()))
        self.assertTrue(math.isnan(pr.probability(condition=x[0])))
        self.assertTrue(math.isnan(pr.probability(x[0], condition=x[0])))

        marginals = pr.marginal_distribution(x)
        self.assertEqual(len(marginals), 3)
        self.assertTrue(math.isnan(marginals[0]))
        self.assertTrue(math.isnan(marginals[1]))
        self.assertTrue(math.isnan(marginals[2]))

        marginals = pr.marginal_distribution(y)
        self.assertEqual(len(marginals), 2)
        self.assertTrue(math.isnan(marginals[0]))
        self.assertTrue(math.isnan(marginals[1]))

        marginals = pr.marginal_distribution(x, y)
        self.assertEqual(len(marginals), 6)
        self.assertTrue(math.isnan(marginals[0]))
        self.assertTrue(math.isnan(marginals[1]))
        self.assertTrue(math.isnan(marginals[2]))
        self.assertTrue(math.isnan(marginals[3]))
        self.assertTrue(math.isnan(marginals[4]))
        self.assertTrue(math.isnan(marginals[5]))


if __name__ == '__main__':
    test_main()
