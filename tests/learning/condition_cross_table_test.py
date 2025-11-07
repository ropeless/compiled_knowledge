from typing import Set

from ck.dataset.cross_table import CrossTable
from ck.learning.condition_cross_table import condition
from ck.pgm import PGM, RandomVariable
from tests.helpers.unittest_fixture import Fixture, test_main


class TestConditionCrossTable(Fixture):

    def test_empty_cross_table(self):
        conditioner = CrossTable(())
        conditioned = CrossTable(())

        reconditioned_crosstab = condition(conditioned, conditioner)

        self.assertEmpty(reconditioned_crosstab.rvs)
        self.assertEmpty(reconditioned_crosstab)

    def test_condition_no_project(self):
        # Create three Boolean random variables
        pgm: PGM = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        c = pgm.new_rv('c', 2)

        cross_table = CrossTable((a, b, c), update=(
            ((0, 0, 0), 1),
            ((0, 0, 1), 2),
            ((0, 1, 0), 3),
            ((0, 1, 1), 4),
            ((1, 0, 0), 5),
            ((1, 0, 1), 6),
            ((1, 1, 0), 7),
            ((1, 1, 1), 8),
        ))
        # weight for a = 0: 1+2+3+4 = 10
        # weight for a = 1: 5+6+7+8 = 26

        conditioner = CrossTable((a,), update=(
            ((0,), 1000),
            ((1,), 52),
        ))

        condition_rvs_set: Set[RandomVariable] = {a}

        reconditioned_crosstab = condition(cross_table, conditioner, condition_rvs_set)

        self.assertEqual(reconditioned_crosstab[(0, 0, 0)], 100)
        self.assertEqual(reconditioned_crosstab[(0, 0, 1)], 200)
        self.assertEqual(reconditioned_crosstab[(0, 1, 0)], 300)
        self.assertEqual(reconditioned_crosstab[(0, 1, 1)], 400)

        self.assertEqual(reconditioned_crosstab[(1, 0, 0)], 10)
        self.assertEqual(reconditioned_crosstab[(1, 0, 1)], 12)
        self.assertEqual(reconditioned_crosstab[(1, 1, 0)], 14)
        self.assertEqual(reconditioned_crosstab[(1, 1, 1)], 16)

    def test_condition_project(self):
        # Create three Boolean random variables
        pgm: PGM = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        c = pgm.new_rv('c', 2)
        d = pgm.new_rv('d', 2)

        cross_table = CrossTable((a, b, c), update=(
            ((0, 0, 0), 1),
            ((0, 0, 1), 2),
            ((0, 1, 0), 3),
            ((0, 1, 1), 4),
            ((1, 0, 0), 5),
            ((1, 0, 1), 6),
            ((1, 1, 0), 7),
            ((1, 1, 1), 8),
        ))
        # weight for a = 0: 1+2+3+4 = 10
        # weight for a = 1: 5+6+7+8 = 26

        conditioner = CrossTable((a, d), update=(
            ((0, 0), 900),
            ((0, 1), 100),
            ((1, 0), 40),
            ((1, 1), 12),
        ))

        condition_rvs_set: Set[RandomVariable] = {a}

        reconditioned_crosstab = condition(cross_table, conditioner, condition_rvs_set)

        self.assertEqual(reconditioned_crosstab[(0, 0, 0)], 100)
        self.assertEqual(reconditioned_crosstab[(0, 0, 1)], 200)
        self.assertEqual(reconditioned_crosstab[(0, 1, 0)], 300)
        self.assertEqual(reconditioned_crosstab[(0, 1, 1)], 400)

        self.assertEqual(reconditioned_crosstab[(1, 0, 0)], 10)
        self.assertEqual(reconditioned_crosstab[(1, 0, 1)], 12)
        self.assertEqual(reconditioned_crosstab[(1, 1, 0)], 14)
        self.assertEqual(reconditioned_crosstab[(1, 1, 1)], 16)

    def test_partial_condition(self):
        # Create three Boolean random variables
        pgm: PGM = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        c = pgm.new_rv('c', 2)

        conditioned = CrossTable((a, b, c), update=(
            ((0, 0, 0), 1),
            ((0, 0, 1), 2),
            ((0, 1, 0), 3),
            ((0, 1, 1), 4),
            ((1, 0, 0), 5),
            ((1, 0, 1), 6),
            ((1, 1, 0), 7),
            ((1, 1, 1), 8),
        ))
        # weight for a = 0: 1+2+3+4 = 10
        # weight for a = 1: 5+6+7+8 = 26

        conditioner = CrossTable((a,), update=(
            ((0,), 1000),
            ((1,), 52),
        ))

        reconditioned_crosstab = condition(conditioned, conditioner)

        self.assertEqual(reconditioned_crosstab[(0, 0, 0)], 100)
        self.assertEqual(reconditioned_crosstab[(0, 0, 1)], 200)
        self.assertEqual(reconditioned_crosstab[(0, 1, 0)], 300)
        self.assertEqual(reconditioned_crosstab[(0, 1, 1)], 400)

        self.assertEqual(reconditioned_crosstab[(1, 0, 0)], 10)
        self.assertEqual(reconditioned_crosstab[(1, 0, 1)], 12)
        self.assertEqual(reconditioned_crosstab[(1, 1, 0)], 14)
        self.assertEqual(reconditioned_crosstab[(1, 1, 1)], 16)

    def test_partial_condition_rvs_provided(self):
        # Create three Boolean random variables
        pgm: PGM = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        c = pgm.new_rv('c', 2)

        conditioned = CrossTable((a, b, c), update=(
            ((0, 0, 0), 1),
            ((0, 0, 1), 2),
            ((0, 1, 0), 3),
            ((0, 1, 1), 4),
            ((1, 0, 0), 5),
            ((1, 0, 1), 6),
            ((1, 1, 0), 7),
            ((1, 1, 1), 8),
        ))
        # weight for a = 0: 1+2+3+4 = 10
        # weight for a = 1: 5+6+7+8 = 26

        conditioner = CrossTable((a,), update=(
            ((0,), 1000),
            ((1,), 52),
        ))

        reconditioned_crosstab = condition(conditioned, conditioner)

        self.assertEqual(reconditioned_crosstab[(0, 0, 0)], 100)
        self.assertEqual(reconditioned_crosstab[(0, 0, 1)], 200)
        self.assertEqual(reconditioned_crosstab[(0, 1, 0)], 300)
        self.assertEqual(reconditioned_crosstab[(0, 1, 1)], 400)

        self.assertEqual(reconditioned_crosstab[(1, 0, 0)], 10)
        self.assertEqual(reconditioned_crosstab[(1, 0, 1)], 12)
        self.assertEqual(reconditioned_crosstab[(1, 1, 0)], 14)
        self.assertEqual(reconditioned_crosstab[(1, 1, 1)], 16)

    def test_condition_with_zeros(self):
        # Create three Boolean random variables
        pgm: PGM = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        c = pgm.new_rv('c', 2)

        cross_table = CrossTable((a, b, c), update=(
            ((0, 0, 0), 1),
            ((0, 0, 1), 2),
            ((0, 1, 0), 3),
            ((0, 1, 1), 4),
            ((1, 0, 0), 5),
            ((1, 0, 1), 6),
            ((1, 1, 0), 7),
            ((1, 1, 1), 8),
        ))
        # weight for a = 0, b = 0: 1+2 = 3
        # weight for a = 1, b = 1: 7+8 = 15

        conditioner = CrossTable((a, b), update=(
            ((0, 0), 300),
            ((1, 1), 30),
        ))

        condition_rvs_set: Set[RandomVariable] = {a, b}

        reconditioned_crosstab = condition(cross_table, conditioner, condition_rvs_set)

        self.assertEqual(reconditioned_crosstab[(0, 0, 0)], 100)
        self.assertEqual(reconditioned_crosstab[(0, 0, 1)], 200)
        self.assertEqual(reconditioned_crosstab[(0, 1, 0)], 0)
        self.assertEqual(reconditioned_crosstab[(0, 1, 1)], 0)

        self.assertEqual(reconditioned_crosstab[(1, 0, 0)], 0)
        self.assertEqual(reconditioned_crosstab[(1, 0, 1)], 0)
        self.assertEqual(reconditioned_crosstab[(1, 1, 0)], 14)
        self.assertEqual(reconditioned_crosstab[(1, 1, 1)], 16)


if __name__ == '__main__':
    test_main()
