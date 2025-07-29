from ck.dataset.cross_table import CrossTable
from ck.learning.coalesce_cross_tables import coalesce_cross_tables
from ck.pgm import PGM
from tests.helpers.unittest_fixture import Fixture, test_main


class TestCoalesceCrossTables(Fixture):

    def test_coalescing_3_cross_table(self):
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)
        z = pgm.new_rv('z', 2)

        # We will make three mutually compatible cross-tables
        # projecting from a master cross-table.
        expect = CrossTable(pgm.rvs)
        expect[0, 0, 0] = 1
        expect[0, 0, 1] = 2
        expect[0, 1, 0] = 3
        expect[0, 1, 1] = 4
        expect[1, 0, 0] = 5
        expect[1, 0, 1] = 6
        expect[1, 1, 0] = 7
        expect[1, 1, 1] = 8

        crosstab_1 = expect.project([x, y])
        crosstab_2 = expect.project([x, z])
        crosstab_3 = expect.project([y, z])

        crosstab: CrossTable = coalesce_cross_tables([crosstab_1, crosstab_2, crosstab_3], pgm.rvs)

        # Convert the expected cross-table to a joint probability distribution
        expect.mul(1 / expect.total_weight())

        self.assertEqual(crosstab.rvs, pgm.rvs)
        self.assertEqual(set(crosstab.keys()), set(expect.keys()))
        for instance in crosstab.keys():
            self.assertAlmostEqual(crosstab[instance], expect[instance])

    def test_coalescing_2_cross_table(self):
        # This is similar to the three cross-table case, but as one is redundant,
        # we should get the same result with just 2 of the 3 cross-tables.
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)
        z = pgm.new_rv('z', 2)

        # We will make three mutually compatible cross-tables
        # projecting from a master cross-table.
        expect = CrossTable(pgm.rvs)
        expect[0, 0, 0] = 1
        expect[0, 0, 1] = 2
        expect[0, 1, 0] = 3
        expect[0, 1, 1] = 4
        expect[1, 0, 0] = 5
        expect[1, 0, 1] = 6
        expect[1, 1, 0] = 7
        expect[1, 1, 1] = 8

        crosstab_1 = expect.project([x, y])
        crosstab_2 = expect.project([x, z])

        crosstab: CrossTable = coalesce_cross_tables([crosstab_1, crosstab_2], pgm.rvs)

        # Convert the expected cross-table to a joint probability distribution
        expect.mul(1 / expect.total_weight())

        self.assertEqual(crosstab.rvs, pgm.rvs)
        self.assertEqual(set(crosstab.keys()), set(expect.keys()))
        for instance in crosstab.keys():
            self.assertAlmostEqual(crosstab[instance], expect[instance])


if __name__ == '__main__':
    test_main()
