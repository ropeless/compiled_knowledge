from ck.dataset import HardDataset
from ck.dataset.cross_table import CrossTable
from ck.dataset.dataset_from_crosstable import dataset_from_cross_table
from ck.pgm import PGM
from tests.helpers.unittest_fixture import Fixture, test_main


class TestDatasetFromCrossTable(Fixture):

    def test_dataset_from_cross_table(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        crosstab: CrossTable = CrossTable(rvs=(x, y))
        crosstab.add((0, 0), 1)
        crosstab.add((1, 0), 5)
        crosstab.add((0, 2), 0.5)

        self.assertEqual(len(crosstab), 3)
        self.assertEqual(crosstab.total_weight(), 6.5)
        self.assertEqual(crosstab.rvs, (x, y))

        dataset: HardDataset = dataset_from_cross_table(crosstab)

        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.total_weight(), 6.5)
        self.assertEqual(dataset.rvs, (x, y))


if __name__ == '__main__':
    test_main()
