from ck.dataset.cross_table import CrossTable
from ck.dataset.cross_table_probabilities import CrossTableProbabilitySpace
from ck.pgm import PGM
from tests.helpers.unittest_fixture import Fixture, test_main


class TestCrossTableProbabilities(Fixture):

    def test_dataset_from_cross_table(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        crosstab: CrossTable = CrossTable(rvs=(x, y))
        crosstab.add((0, 0), 1)
        crosstab.add((1, 0), 5)
        crosstab.add((0, 2), 2)
        crosstab.add((1, 1), 2)

        pr = CrossTableProbabilitySpace(crosstab)

        self.assertEqual(pr.z, 10)
        self.assertEqual(pr.rvs, (x, y))

        self.assertEqual(pr.wmc(x[0], y[0]), 1)
        self.assertEqual(pr.wmc(x[0], y[1]), 0)
        self.assertEqual(pr.wmc(x[0], y[2]), 2)
        self.assertEqual(pr.wmc(x[1], y[0]), 5)
        self.assertEqual(pr.wmc(x[1], y[1]), 2)
        self.assertEqual(pr.wmc(x[1], y[2]), 0)
        self.assertEqual(pr.wmc(x[0]), 3)
        self.assertEqual(pr.wmc(x[1]), 7)
        self.assertEqual(pr.wmc(y[0]), 6)
        self.assertEqual(pr.wmc(y[1]), 2)
        self.assertEqual(pr.wmc(y[2]), 2)
        self.assertEqual(pr.wmc(), 10)

        self.assertEqual(pr.probability(x[0], y[0]), 0.1)
        self.assertEqual(pr.probability(x[0], y[1]), 0.0)
        self.assertEqual(pr.probability(x[0], y[2]), 0.2)
        self.assertEqual(pr.probability(x[1], y[0]), 0.5)
        self.assertEqual(pr.probability(x[1], y[1]), 0.2)
        self.assertEqual(pr.probability(x[1], y[2]), 0.0)
        self.assertEqual(pr.probability(x[0]), 0.3)
        self.assertEqual(pr.probability(x[1]), 0.7)
        self.assertEqual(pr.probability(y[0]), 0.6)
        self.assertEqual(pr.probability(y[1]), 0.2)
        self.assertEqual(pr.probability(y[2]), 0.2)
        self.assertEqual(pr.probability(), 1.0)


if __name__ == '__main__':
    test_main()
