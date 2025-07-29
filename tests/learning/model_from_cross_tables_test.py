from ck.dataset.cross_table import CrossTable
from ck.learning.model_from_cross_tables import model_from_cross_tables
from ck.pgm import PGM, PotentialFunction
from tests.helpers.unittest_fixture import Fixture, test_main


class TestModelFromCrossTables(Fixture):

    def test_empty_pgm(self):
        pgm: PGM = PGM()

        model_from_cross_tables(pgm, [])

        self.assertEmpty(pgm.rvs)
        self.assertTrue(pgm.check_is_bayesian_network())
        self.assertEmpty(pgm.factors)

    def test_no_cross_tables(self):
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 3)

        model_from_cross_tables(pgm, [])

        self.assertArrayEqual(pgm.rvs, (x, y))
        self.assertTrue(pgm.check_is_bayesian_network())

        self.assertEqual(len(pgm.factors), 2)
        self.assertTrue(pgm.factors[0].is_zero)
        self.assertTrue(pgm.factors[1].is_zero)

    def test_one_cross_table(self):
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 3)

        crosstab = CrossTable((x, y))
        crosstab[0, 0] = 1
        crosstab[0, 1] = 2
        crosstab[0, 2] = 3
        crosstab[1, 0] = 4
        crosstab[1, 1] = 5
        crosstab[1, 2] = 6

        model_from_cross_tables(pgm, [crosstab])

        self.assertArrayEqual(pgm.rvs, (x, y))
        self.assertTrue(pgm.check_is_bayesian_network())

        self.assertEqual(len(pgm.factors), 2)
        # We have no guarantee of the structure, it could be either:
        # a. (x), (y, x)
        # b. (y, x), (x)
        # c. (y), (x, y)
        # d. (x, y), (y)
        if pgm.factors[0].rvs[0] == x:
            # case a or d
            factor_x, factor_y = pgm.factors
        else:
            # case b or c
            factor_y, factor_x = pgm.factors
        f_x: PotentialFunction = factor_x.function
        f_y: PotentialFunction = factor_y.function
        total = crosstab.total_weight()

        if len(factor_x.rvs) == 0:
            # case a or b, x is the parent, y is the child
            self.assertAlmostEqual(f_x[0], (1 + 2 + 3) / total)
            self.assertAlmostEqual(f_x[1], (4 + 5 + 6) / total)

            self.assertAlmostEqual(f_y[0, 0], 1 / (1 + 2 + 3))
            self.assertAlmostEqual(f_y[1, 0], 2 / (1 + 2 + 3))
            self.assertAlmostEqual(f_y[2, 0], 3 / (1 + 2 + 3))
            self.assertAlmostEqual(f_y[0, 1], 4 / (4 + 5 + 6))
            self.assertAlmostEqual(f_y[1, 1], 5 / (4 + 5 + 6))
            self.assertAlmostEqual(f_y[2, 1], 6 / (4 + 5 + 6))

        else:
            # case c or d, y is the parent, x is the child
            self.assertAlmostEqual(f_y[0], (1 + 4) / total)
            self.assertAlmostEqual(f_y[1], (2 + 5) / total)
            self.assertAlmostEqual(f_y[2], (3 + 6) / total)

            self.assertAlmostEqual(f_x[0, 0], 1 / (1 + 4))
            self.assertAlmostEqual(f_x[0, 1], 2 / (2 + 5))
            self.assertAlmostEqual(f_x[0, 2], 3 / (3 + 6))
            self.assertAlmostEqual(f_x[1, 0], 4 / (1 + 4))
            self.assertAlmostEqual(f_x[1, 1], 5 / (2 + 5))
            self.assertAlmostEqual(f_x[1, 2], 6 / (3 + 6))

    def test_coalescing_cross_table(self):
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)
        z = pgm.new_rv('z', 2)

        # We will make three mutually compatible cross-tables
        # projecting from a master cross-table.
        master_crosstab = CrossTable((x, y, z))
        master_crosstab[0, 0, 0] = 1
        master_crosstab[0, 0, 1] = 2
        master_crosstab[0, 1, 0] = 3
        master_crosstab[0, 1, 1] = 4
        master_crosstab[1, 0, 0] = 5
        master_crosstab[1, 0, 1] = 6
        master_crosstab[1, 1, 0] = 7
        master_crosstab[1, 1, 1] = 8

        crosstab_1 = master_crosstab.project([x, y])
        crosstab_2 = master_crosstab.project([x, z])
        crosstab_3 = master_crosstab.project([y, z])

        model_from_cross_tables(pgm, [crosstab_1, crosstab_2, crosstab_3])


if __name__ == '__main__':
    test_main()
