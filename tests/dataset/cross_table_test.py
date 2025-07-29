from typing import Sequence, List, Tuple

import numpy as np

from ck.dataset import SoftDataset, HardDataset
from ck.dataset.cross_table import CrossTable, cross_table_from_hard_dataset, cross_table_from_soft_dataset, \
    Instance, cross_table_from_dataset
from ck.pgm import PGM, rv_instances
from ck.utils.iter_extras import multiply
from tests.helpers.unittest_fixture import Fixture, test_main


class TestCrossTable(Fixture):
    def test_empty(self):
        crosstab: CrossTable = CrossTable(rvs=())

        self.assertEqual(crosstab.rvs, ())
        self.assertEqual(len(crosstab), 0)
        self.assertEqual(crosstab.total_weight(), 0)

    def test_no_instances(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        crosstab: CrossTable = CrossTable(rvs=(x, y))

        self.assertEqual(crosstab.rvs, (x, y))
        self.assertEqual(len(crosstab), 0)
        self.assertEqual(crosstab.total_weight(), 0)

        for instance in rv_instances(x, y):
            self.assertEqual(crosstab[tuple(instance)], 0.0)

    def test_update_in_constructor(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        update: List[Tuple[Instance, float]] = [
            ((0, 0), 1.0),
            ((0, 2), 1.0),
            ((1, 1), 2.0),
            ((0, 0), 1.0),
            ((1, 0), 1.0),
        ]
        crosstab: CrossTable = CrossTable(rvs=(x, y), update=update)

        self.assertEqual(crosstab.rvs, (x, y))
        self.assertEqual(len(crosstab), 4)
        self.assertAlmostEqual(crosstab.total_weight(), 6.0)

        self.assertEqual(crosstab[(0, 0)], 2.0)
        self.assertEqual(crosstab[(0, 1)], 0.0)
        self.assertEqual(crosstab[(0, 2)], 1.0)
        self.assertEqual(crosstab[(1, 0)], 1.0)
        self.assertEqual(crosstab[(1, 1)], 2.0)
        self.assertEqual(crosstab[(1, 2)], 0.0)

    def test_dirichlet_prior_float(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        crosstab: CrossTable = CrossTable(rvs=(x, y), dirichlet_prior=0.1)

        self.assertEqual(crosstab.rvs, (x, y))
        self.assertEqual(len(crosstab), 6)
        self.assertAlmostEqual(crosstab.total_weight(), 0.6)

        for instance in rv_instances(x, y):
            self.assertEqual(crosstab[tuple(instance)], 0.1)

    def test_dirichlet_prior_cross_table(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        prior = CrossTable(rvs=(y, x))  # Note the rvs are in reverse order!
        prior[(0, 0)] = 1.0
        prior[(0, 1)] = 2.0
        prior[(2, 1)] = 3.0

        crosstab: CrossTable = CrossTable(rvs=(x, y), dirichlet_prior=prior)

        self.assertEqual(crosstab.rvs, (x, y))
        self.assertEqual(len(crosstab), 3)
        self.assertAlmostEqual(crosstab.total_weight(), 6.0)

        self.assertEqual(crosstab[(0, 0)], 1.0)
        self.assertEqual(crosstab[(1, 0)], 2.0)
        self.assertEqual(crosstab[(1, 2)], 3.0)

    def test_update_and_dirichlet_prior_in_constructor(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        update: List[Tuple[Instance, float]] = [
            ((0, 0), 1.0),
            ((0, 2), 1.0),
            ((1, 1), 2.0),
            ((0, 0), 1.0),
            ((1, 0), 1.0),
        ]
        crosstab: CrossTable = CrossTable(rvs=(x, y), update=update, dirichlet_prior=0.1)

        self.assertEqual(crosstab.rvs, (x, y))
        self.assertEqual(len(crosstab), 6)
        self.assertAlmostEqual(crosstab.total_weight(), 6.6)

        self.assertEqual(crosstab[(0, 0)], 2.1)
        self.assertEqual(crosstab[(0, 1)], 0.1)
        self.assertEqual(crosstab[(0, 2)], 1.1)
        self.assertEqual(crosstab[(1, 0)], 1.1)
        self.assertEqual(crosstab[(1, 1)], 2.1)
        self.assertEqual(crosstab[(1, 2)], 0.1)

    def test_zero_weights(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        crosstab: CrossTable = CrossTable(rvs=(x, y))

        self.assertEqual(len(crosstab), 0)
        self.assertEqual(sorted(crosstab.items()), [])
        self.assertEqual(crosstab[0, 0], 0)

        crosstab[0, 0] = 1
        crosstab[0, 1] = 2
        crosstab[0, 2] = 0  # should be ignored
        crosstab[1, 2] = 3

        self.assertEqual(len(crosstab), 3)
        self.assertEqual(sorted(crosstab.items()), [((0, 0), 1), ((0, 1), 2), ((1, 2), 3), ])
        self.assertEqual(crosstab[0, 2], 0)

        crosstab[0, 1] = 0

        self.assertEqual(len(crosstab), 2)
        self.assertEqual(sorted(crosstab.items()), [((0, 0), 1), ((1, 2), 3), ])
        self.assertEqual(crosstab[0, 1], 0)

        crosstab.add((1, 2), -3)

        self.assertEqual(len(crosstab), 1)
        self.assertEqual(crosstab[1, 2], 0)

        self.assertEqual(sorted(crosstab.items()), [((0, 0), 1), ])

        del crosstab[0, 0]

        self.assertEqual(len(crosstab), 0)
        self.assertEqual(sorted(crosstab.items()), [])
        self.assertEqual(crosstab[0, 0], 0)

    def test_eq(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)
        z = pgm.new_rv('z', 2)

        crosstab_1: CrossTable = CrossTable(rvs=(x, y), update=[((0, 0), 1), ((1, 2), 3), ])
        crosstab_2: CrossTable = CrossTable(rvs=(x, y), update=[((0, 0), 1), ((1, 2), 3), ])
        crosstab_3: CrossTable = CrossTable(rvs=(x, y), update=[((0, 0), 1), ])
        crosstab_4: CrossTable = CrossTable(rvs=(x, z), update=[((0, 0), 1), ((1, 2), 3), ])

        self.assertEqual(crosstab_1, crosstab_2)
        self.assertNotEqual(crosstab_1, crosstab_3)
        self.assertNotEqual(crosstab_1, crosstab_4)

    def test_cross_table_from_dataset(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        hard_dataset = HardDataset([
            (x, [0, 0, 1, 1, 0, 1]),
            (y, [0, 2, 1, 1, 0, 1])
        ])

        x_state_weights = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        y_state_weights = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        soft_dataset = SoftDataset([
            (x, x_state_weights),
            (y, y_state_weights),
        ])

        crosstab: CrossTable = cross_table_from_dataset(hard_dataset)
        expect = {
            (0, 0): 2,
            (0, 2): 1,
            (1, 1): 3,
        }
        self.assertEqual(dict(crosstab), expect)

        crosstab: CrossTable = cross_table_from_dataset(soft_dataset)
        expect = {
            (0, 0): 2,
            (0, 2): 1,
            (1, 1): 3,
        }
        self.assertDictAlmostEqual(crosstab, expect)


class TestCrossTableFromHard(Fixture):
    def test_empty(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = HardDataset([
            (x, [0, 0, 1, 1, 0, 1]),
            (y, [0, 2, 1, 1, 0, 1])
        ])

        crosstab: CrossTable = cross_table_from_hard_dataset(dataset, ())
        self.assertEqual(crosstab.rvs, ())
        expect = {
            (): 6,
        }
        self.assertEqual(dict(crosstab), expect)
        self.assertEqual(crosstab.total_weight(), 6)

    def test_one_rv(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = HardDataset([
            (x, [0, 0, 1, 1, 0, 1]),
            (y, [0, 2, 1, 1, 0, 1])
        ])

        crosstab_x: CrossTable = cross_table_from_hard_dataset(dataset, [x])
        self.assertEqual(crosstab_x.rvs, (x,))
        expect = {
            (0,): 3,
            (1,): 3,
        }
        self.assertEqual(dict(crosstab_x), expect)
        self.assertEqual(crosstab_x.total_weight(), 6)

        crosstab_y: CrossTable = cross_table_from_hard_dataset(dataset, [y])
        self.assertEqual(crosstab_y.rvs, (y,))
        expect = {
            (0,): 2,
            (1,): 3,
            (2,): 1,
        }
        self.assertEqual(dict(crosstab_y), expect)
        self.assertEqual(crosstab_y.total_weight(), 6)

    def test_two_rvs(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = HardDataset([
            (x, [0, 0, 1, 1, 0, 1]),
            (y, [0, 2, 1, 1, 0, 1])
        ])

        crosstab: CrossTable = cross_table_from_hard_dataset(dataset, [x, y])
        self.assertEqual(crosstab.rvs, (x, y))
        expect = {
            (0, 0): 2,
            (0, 2): 1,
            (1, 1): 3,
        }
        self.assertEqual(dict(crosstab), expect)
        self.assertEqual(crosstab.total_weight(), 6)

    def test_weighted(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        dataset = HardDataset(
            [
                (x, [0, 0, 1, 1, 0, 1]),
                (y, [1, 2, 0, 0, 1, 0])
            ],
            weights=[1, 0, 3, 1, 1, 1],
        )

        crosstab_x: CrossTable = cross_table_from_hard_dataset(dataset, [x])
        expect = {
            (0,): 2,
            (1,): 5,
        }
        self.assertEqual(dict(crosstab_x), expect)
        self.assertEqual(crosstab_x.total_weight(), 7)

        crosstab_y: CrossTable = cross_table_from_hard_dataset(dataset, [y])
        expect = {
            (0,): 5,
            (1,): 2,
        }
        self.assertEqual(dict(crosstab_y), expect)
        self.assertEqual(crosstab_y.total_weight(), 7)


class TestCrossTableFromSoft(Fixture):

    def test_empty(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        x_state_weights = np.array([
            [0.6, 0.4],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.3, 0.7],
            [0.9, 0.1],
        ])
        y_state_weights = np.array([
            [0.6, 0.3, 0.1],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.8, 0.1, 0.1],
        ])

        dataset = SoftDataset([
            (x, x_state_weights),
            (y, y_state_weights),
        ])

        crosstab: CrossTable = cross_table_from_soft_dataset(dataset, ())
        self.assertEqual(crosstab.rvs, ())
        expect = {
            (): 5,
        }
        self.assertEqual(dict(crosstab), expect)
        self.assertEqual(crosstab.total_weight(), 5)

    def test_one_rv(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        x_state_weights = np.array([
            [0.6, 0.4],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.3, 0.7],
            [0.9, 0.1],
        ])
        y_state_weights = np.array([
            [0.6, 0.3, 0.1],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.8, 0.1, 0.1],
        ])

        dataset = SoftDataset([
            (x, x_state_weights),
            (y, y_state_weights),
        ])

        crosstab_x: CrossTable = cross_table_from_soft_dataset(dataset, [x])
        self.assertEqual(crosstab_x.rvs, (x,))
        expect = {
            (0,): 0.6 + 1.0 + 0.0 + 0.3 + 0.9,
            (1,): 0.4 + 0.0 + 1.0 + 0.7 + 0.1,
        }
        self.assertDictAlmostEqual(crosstab_x, expect)
        self.assertEqual(crosstab_x.total_weight(), 5)

        crosstab_y: CrossTable = cross_table_from_soft_dataset(dataset, [y])
        self.assertEqual(crosstab_y.rvs, (y,))
        expect = {
            (0,): 0.6 + 0.0 + 0.0 + 0.3 + 0.8,
            (1,): 0.3 + 0.0 + 1.0 + 0.4 + 0.1,
            (2,): 0.1 + 1.0 + 0.0 + 0.3 + 0.1,
        }
        self.assertDictAlmostEqual(crosstab_y, expect)
        self.assertEqual(crosstab_y.total_weight(), 5)

    def test_two_rvs(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        x_state_weights = np.array([
            [0.6, 0.4],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.3, 0.7],
            [0.9, 0.1],
        ])
        y_state_weights = np.array([
            [0.6, 0.3, 0.1],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.8, 0.1, 0.1],
        ])

        dataset = SoftDataset([
            (x, x_state_weights),
            (y, y_state_weights),
        ])

        crosstab: CrossTable = cross_table_from_soft_dataset(dataset, [x, y])
        self.assertEqual(crosstab.rvs, (x, y))
        expect = {
            (0, 0): _sum_product((0.6, 1.0, 0.0, 0.3, 0.9), (0.6, 0.0, 0.0, 0.3, 0.8)),
            (0, 1): _sum_product((0.6, 1.0, 0.0, 0.3, 0.9), (0.3, 0.0, 1.0, 0.4, 0.1)),
            (0, 2): _sum_product((0.6, 1.0, 0.0, 0.3, 0.9), (0.1, 1.0, 0.0, 0.3, 0.1)),
            (1, 0): _sum_product((0.4, 0.0, 1.0, 0.7, 0.1), (0.6, 0.0, 0.0, 0.3, 0.8)),
            (1, 1): _sum_product((0.4, 0.0, 1.0, 0.7, 0.1), (0.3, 0.0, 1.0, 0.4, 0.1)),
            (1, 2): _sum_product((0.4, 0.0, 1.0, 0.7, 0.1), (0.1, 1.0, 0.0, 0.3, 0.1)),
        }
        self.assertDictAlmostEqual(crosstab, expect)
        self.assertEqual(crosstab.total_weight(), 5)

    def test_weighted(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        x_state_weights = np.array([
            [0.6, 0.4],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.3, 0.7],
            [0.9, 0.1],
        ])
        y_state_weights = np.array([
            [0.6, 0.3, 0.1],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.3, 0.4, 0.3],
            [0.8, 0.1, 0.1],
        ])
        weights = [1, 0, 3, 1, 1]

        dataset = SoftDataset(
            [
                (x, x_state_weights),
                (y, y_state_weights),
            ],
            weights=weights,
        )

        crosstab_x: CrossTable = cross_table_from_soft_dataset(dataset, [x])
        self.assertEqual(crosstab_x.rvs, (x,))
        expect = {
            (0,): _sum_product((0.6, 1.0, 0.0, 0.3, 0.9), weights),
            (1,): _sum_product((0.4, 0.0, 1.0, 0.7, 0.1), weights),
        }
        self.assertDictAlmostEqual(crosstab_x, expect)
        self.assertAlmostEqual(crosstab_x.total_weight(), 6)

        crosstab_y: CrossTable = cross_table_from_soft_dataset(dataset, [y])
        self.assertEqual(crosstab_y.rvs, (y,))
        expect = {
            (0,): _sum_product((0.6, 0.0, 0.0, 0.3, 0.8), weights),
            (1,): _sum_product((0.3, 0.0, 1.0, 0.4, 0.1), weights),
            (2,): _sum_product((0.1, 1.0, 0.0, 0.3, 0.1), weights),
        }
        self.assertDictAlmostEqual(crosstab_y, expect)
        self.assertAlmostEqual(crosstab_y.total_weight(), 6)


def _sum_product(*vals: Sequence[float]) -> float:
    """
    Multiply each corresponding element of the given sequences,
    then sum up those products.

    e.g., _sum_product([a, b], [c, d]) = a * b + c * d.
    """
    return sum(
        multiply(to_product)
        for to_product in zip(*vals)
    )


if __name__ == '__main__':
    test_main()
