import math
from typing import Sequence, Iterable, Tuple, Dict

from ck.dataset.cross_table import CrossTable
from ck.pgm import RandomVariable, Instance, PGM
from ck.probability.cross_table_probability_space import CrossTableProbabilitySpace
from ck.probability.probability_space import ProbabilitySpace, Condition, plogp
from tests.helpers.unittest_fixture import Fixture, test_main


class DummyProbabilitySpace(ProbabilitySpace):
    """
    We need to instantiate a Probability space for testing.
    The challenge is to implement the `wmc` method.
    Here we use `CrossTableProbabilitySpace` which has a simple implementation of `wmc`.
    """

    def __init__(
            self,
            rvs: Sequence[RandomVariable],
            update: Iterable[Tuple[Instance, float]] = (),
    ):
        cross_table: CrossTable = CrossTable(rvs, update=update)
        self._rvs = cross_table.rvs
        self._wmc = CrossTableProbabilitySpace(cross_table)
        self._z = cross_table.total_weight()

    @property
    def rvs(self) -> Sequence[RandomVariable]:
        return self._rvs

    def wmc(self, *condition: Condition) -> float:
        return self._wmc.wmc(*condition)

    @property
    def z(self) -> float:
        return self._z


class TestProbabilitySpace(Fixture):

    def test_no_rv(self):
        space = DummyProbabilitySpace(rvs=())

        self.assertArrayEqual(space.rvs, [])
        self.assertEqual(space.z, 0)
        self.assertEqual(space.wmc(), 0)
        self.assertNan(space.probability())
        self.assertArrayEqual(space.marginal_distribution(), [math.nan], nan_equality=True)

        pr, instance = space.map()
        self.assertNan(pr)
        self.assertArrayEqual(instance, ())

    def test_one_rv(self):
        pgm: PGM = PGM()
        x: RandomVariable = pgm.new_rv('x', 3)
        distribution: Dict[Instance, float] = {
            (0,): 2,
            (1,): 3,
            (2,): 5,
        }
        space = DummyProbabilitySpace(pgm.rvs, distribution.items())

        self.assertArrayEqual(space.rvs, pgm.rvs)
        self.assertEqual(space.z, 10)

        self.assertEqual(space.wmc(), 10)
        self.assertEqual(space.wmc(x[0]), 2)
        self.assertEqual(space.wmc(x[1]), 3)
        self.assertEqual(space.wmc(x[2]), 5)

        self.assertAlmostEqual(space.probability(), 1.0)
        self.assertAlmostEqual(space.probability(x[0]), 0.2)
        self.assertAlmostEqual(space.probability(x[1]), 0.3)
        self.assertAlmostEqual(space.probability(x[2]), 0.5)

        self.assertArrayEqual(space.marginal_distribution(), [1])
        self.assertArrayEqual(space.marginal_distribution(x), [0.2, 0.3, 0.5])

        pr, instance = space.map()
        self.assertAlmostEqual(pr, 1)
        self.assertArrayEqual(instance, ())

        pr, instance = space.map(x)
        self.assertAlmostEqual(pr, 0.5)
        self.assertArrayEqual(instance, (2,))

        self.assertAlmostEqual(space.entropy(x), - plogp(0.2) - plogp(0.3) - plogp(0.5))
        self.assertAlmostEqual(space.entropy(x, condition=x[2]), 0)
        self.assertAlmostEqual(space.entropy(x, condition=(x[0], x[1])), - plogp(0.4) - plogp(0.6))

    def test_two_rvs(self):
        pgm: PGM = PGM()
        x: RandomVariable = pgm.new_rv('x', 3)
        y: RandomVariable = pgm.new_rv('y', 3)
        distribution: Dict[Instance, float] = {
            (0, 0): 2,
            (1, 0): 3,
            (2, 0): 5,

            (0, 1): 20,
            (1, 1): 30,
            (2, 1): 50,

            (0, 2): 200,
            (1, 2): 300,
            (2, 2): 500,
        }
        space = DummyProbabilitySpace(pgm.rvs, distribution.items())

        self.assertArrayEqual(space.rvs, pgm.rvs)
        self.assertEqual(space.z, 1110)

        self.assertEqual(space.wmc(), 1110)

        self.assertEqual(space.wmc(x[0]), 222)
        self.assertEqual(space.wmc(x[1]), 333)
        self.assertEqual(space.wmc(x[2]), 555)

        self.assertEqual(space.wmc(y[0]), 10)
        self.assertEqual(space.wmc(y[1]), 100)
        self.assertEqual(space.wmc(y[2]), 1000)

        self.assertEqual(space.wmc(x[0], y[0]), 2)
        self.assertEqual(space.wmc(x[1], y[0]), 3)
        self.assertEqual(space.wmc(x[2], y[0]), 5)

        self.assertEqual(space.wmc(x[0], y[1]), 20)
        self.assertEqual(space.wmc(x[1], y[1]), 30)
        self.assertEqual(space.wmc(x[2], y[1]), 50)

        self.assertEqual(space.wmc(x[0], y[2]), 200)
        self.assertEqual(space.wmc(x[1], y[2]), 300)
        self.assertEqual(space.wmc(x[2], y[2]), 500)

        self.assertAlmostEqual(space.probability(), 1.0)

        self.assertAlmostEqual(space.probability(x[0]), 222 / 1110)
        self.assertAlmostEqual(space.probability(x[1]), 333 / 1110)
        self.assertAlmostEqual(space.probability(x[2]), 555 / 1110)

        self.assertAlmostEqual(space.probability(y[0]), 10 / 1110)
        self.assertAlmostEqual(space.probability(y[1]), 100 / 1110)
        self.assertAlmostEqual(space.probability(y[2]), 1000 / 1110)

        self.assertAlmostEqual(space.probability(x[0], y[0]), 2 / 1110)
        self.assertAlmostEqual(space.probability(x[1], y[0]), 3 / 1110)
        self.assertAlmostEqual(space.probability(x[2], y[0]), 5 / 1110)

        self.assertAlmostEqual(space.probability(x[0], y[1]), 20 / 1110)
        self.assertAlmostEqual(space.probability(x[1], y[1]), 30 / 1110)
        self.assertAlmostEqual(space.probability(x[2], y[1]), 50 / 1110)

        self.assertAlmostEqual(space.probability(x[0], y[2]), 200 / 1110)
        self.assertAlmostEqual(space.probability(x[1], y[2]), 300 / 1110)
        self.assertAlmostEqual(space.probability(x[2], y[2]), 500 / 1110)

        self.assertAlmostEqual(space.probability(x[0], condition=y[0]), 2 / 10)
        self.assertAlmostEqual(space.probability(x[0], condition=y[1]), 20 / 100)
        self.assertAlmostEqual(space.probability(x[0], condition=y[2]), 200 / 1000)

        self.assertAlmostEqual(space.probability(x[1], condition=y[0]), 3 / 10)
        self.assertAlmostEqual(space.probability(x[1], condition=y[1]), 30 / 100)
        self.assertAlmostEqual(space.probability(x[1], condition=y[2]), 300 / 1000)

        self.assertAlmostEqual(space.probability(x[2], condition=y[0]), 5 / 10)
        self.assertAlmostEqual(space.probability(x[2], condition=y[1]), 50 / 100)
        self.assertAlmostEqual(space.probability(x[2], condition=y[2]), 500 / 1000)

        self.assertAlmostEqual(space.probability(y[0], condition=x[0]), 2 / 222)
        self.assertAlmostEqual(space.probability(y[0], condition=x[1]), 3 / 333)
        self.assertAlmostEqual(space.probability(y[0], condition=x[2]), 5 / 555)

        self.assertAlmostEqual(space.probability(y[1], condition=x[0]), 20 / 222)
        self.assertAlmostEqual(space.probability(y[1], condition=x[1]), 30 / 333)
        self.assertAlmostEqual(space.probability(y[1], condition=x[2]), 50 / 555)

        self.assertAlmostEqual(space.probability(y[2], condition=x[0]), 200 / 222)
        self.assertAlmostEqual(space.probability(y[2], condition=x[1]), 300 / 333)
        self.assertAlmostEqual(space.probability(y[2], condition=x[2]), 500 / 555)

        self.assertAlmostEqual(space.probability(x[0], y[0], condition=x[0]), 2 / 222)
        self.assertAlmostEqual(space.probability(x[1], y[0], condition=x[0]), 0)
        self.assertAlmostEqual(space.probability(x[0], y[0], condition=x[1]), 0)
        self.assertAlmostEqual(space.probability(x[1], y[0], condition=x[1]), 3 / 333)

        self.assertAlmostEqual(space.probability(x[0], x[1], y[0], condition=x[0]), 2 / 222)
        self.assertAlmostEqual(space.probability(x[0], x[1], y[0], condition=x[1]), 3 / 333)

        self.assertAlmostEqual(space.probability(x[0], condition=(x[0], x[1])), 222 / (222 + 333))
        self.assertAlmostEqual(space.probability(x[1], condition=(x[0], x[1])), 333 / (222 + 333))

        self.assertAlmostEqual(space.probability(x[0], y[0], condition=(x[0], x[1])), 2 / (222 + 333))
        self.assertAlmostEqual(space.probability(x[1], y[0], condition=(x[0], x[1])), 3 / (222 + 333))

    def test_marginal_distribution(self):
        pgm: PGM = PGM()
        x: RandomVariable = pgm.new_rv('x', 3)
        y: RandomVariable = pgm.new_rv('y', 3)
        distribution: Dict[Instance, float] = {
            (0, 0): 2,
            (1, 0): 3,
            (2, 0): 5,

            (0, 1): 20,
            (1, 1): 30,
            (2, 1): 50,

            (0, 2): 200,
            (1, 2): 300,
            (2, 2): 500,
        }
        space = DummyProbabilitySpace(pgm.rvs, distribution.items())

        self.assertArrayEqual(space.marginal_distribution(), [1])
        self.assertArrayEqual(space.marginal_distribution(x), [222 / 1110, 333 / 1110, 555 / 1110])
        self.assertArrayEqual(space.marginal_distribution(y), [10 / 1110, 100 / 1110, 1000 / 1110])
        self.assertArrayEqual(space.marginal_distribution(x, y), [
            2 / 1110, 20 / 1110, 200 / 1110,
            3 / 1110, 30 / 1110, 300 / 1110,
            5 / 1110, 50 / 1110, 500 / 1110,
        ])

        self.assertArrayEqual(space.marginal_distribution(x, condition=x[1]), [0, 1, 0])
        self.assertArrayEqual(space.marginal_distribution(x, condition=(x[0], x[2])), [222 / 777, 0, 555 / 777])
        self.assertArrayEqual(space.marginal_distribution(x, condition=y[0]), [2 / 10, 3 / 10, 5 / 10])
        self.assertArrayEqual(space.marginal_distribution(x, condition=(x[0], x[2], y[1])), [20 / 70, 0, 50 / 70])

    def test_map(self):
        pgm: PGM = PGM()
        x: RandomVariable = pgm.new_rv('x', 3)
        y: RandomVariable = pgm.new_rv('y', 3)
        distribution: Dict[Instance, float] = {
            (0, 0): 2,
            (1, 0): 3,
            (2, 0): 5,

            (0, 1): 20,
            (1, 1): 30,
            (2, 1): 50,

            (0, 2): 200,
            (1, 2): 300,
            (2, 2): 500,
        }
        space = DummyProbabilitySpace(pgm.rvs, distribution.items())

        pr, instance = space.map()
        self.assertAlmostEqual(pr, 1)
        self.assertArrayEqual(instance, ())

        pr, instance = space.map(x)
        self.assertAlmostEqual(pr, 555 / 1110)
        self.assertArrayEqual(instance, (2,))

        pr, instance = space.map(y)
        self.assertAlmostEqual(pr, 1000 / 1110)
        self.assertArrayEqual(instance, (2,))

        pr, instance = space.map(x, y)
        self.assertAlmostEqual(pr, 500 / 1110)
        self.assertArrayEqual(instance, (2, 2))

        # TODO - conditioned map

    def test_correlation(self):
        pgm: PGM = PGM()
        x: RandomVariable = pgm.new_rv('x', 3)
        y: RandomVariable = pgm.new_rv('y', 3)
        distribution: Dict[Instance, float] = {
            (0, 0): 2,
            (1, 0): 3,
            (2, 0): 5,

            (0, 1): 20,
            (1, 1): 30,
            (2, 1): 50,

            (0, 2): 200,
            (1, 2): 300,
            (2, 2): 500,
        }
        space = DummyProbabilitySpace(pgm.rvs, distribution.items())

        self.assertAlmostEqual(space.correlation(x[0], x[0]), 1)
        self.assertAlmostEqual(space.correlation(x[0], x[1]), 0)
        self.assertAlmostEqual(space.correlation(x[0], x[2]), 0)

        r = space.correlation(x[0], y[0])
        p1 = 222 / 1110
        p2 = 10 / 1110
        p12 = 2 / 1110
        expect = (p12 - p1 * p2) / math.sqrt(p1 * (1.0 - p1) * p2 * (1.0 - p2))
        self.assertAlmostEqual(r, expect)

        r = space.correlation(x[0], y[0], condition=x[0])
        self.assertAlmostEqual(r, 0)

        r = space.correlation(x[0], y[0], condition=x[1])
        self.assertAlmostEqual(r, 0)

    def test_entropy(self):
        pgm: PGM = PGM()
        x: RandomVariable = pgm.new_rv('x', 3)
        y: RandomVariable = pgm.new_rv('y', 3)
        distribution: Dict[Instance, float] = {
            (0, 0): 2,
            (1, 0): 3,
            (2, 0): 5,

            (0, 1): 20,
            (1, 1): 30,
            (2, 1): 50,

            (0, 2): 200,
            (1, 2): 300,
            (2, 2): 500,
        }
        space = DummyProbabilitySpace(pgm.rvs, distribution.items())

        self.assertAlmostEqual(space.entropy(x), - plogp(222 / 1110) - plogp(333 / 1110) - plogp(555 / 1110))
        self.assertAlmostEqual(space.entropy(x, condition=x[2]), 0)
        self.assertAlmostEqual(space.entropy(x, condition=(x[0], x[1])), - plogp(222 / 555) - plogp(333 / 555))

        # TODO test conditional

    def test_joint_entropy(self):
        pgm: PGM = PGM()
        x: RandomVariable = pgm.new_rv('x', 3)
        y: RandomVariable = pgm.new_rv('y', 3)
        distribution: Dict[Instance, float] = {
            (0, 0): 2,
            (1, 0): 3,
            (2, 0): 5,

            (0, 1): 20,
            (1, 1): 30,
            (2, 1): 50,

            (0, 2): 200,
            (1, 2): 300,
            (2, 2): 500,
        }
        space = DummyProbabilitySpace(pgm.rvs, distribution.items())

        expect = (
                - plogp(2 / 1110) - plogp(3 / 1110) - plogp(5 / 1110)
                - plogp(20 / 1110) - plogp(30 / 1110) - plogp(50 / 1110)
                - plogp(200 / 1110) - plogp(300 / 1110) - plogp(500 / 1110)
        )
        self.assertAlmostEqual(space.joint_entropy(x, y), expect)

        self.assertAlmostEqual(space.joint_entropy(x, x), space.entropy(x))
        self.assertAlmostEqual(space.joint_entropy(y, y), space.entropy(y))

        # TODO test conditional


if __name__ == '__main__':
    test_main()
