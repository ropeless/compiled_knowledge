from typing import Dict

from ck.dataset import HardDataset, SoftDataset
from ck.dataset.cross_table import CrossTable
from ck.dataset.dataset_from_crosstable import dataset_from_cross_table, expand_soft_dataset
from ck.pgm import PGM, Instance
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

    def test_expand_soft_dataset(self):
        pgm = PGM()
        x = pgm.new_rv('x', 3)
        y = pgm.new_rv('y', 3)

        weights = [0.0, 2.0, 3.0, 5.0]
        x_data = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.1, 0.8, 0.0],
            [0.2, 0.0, 0.3],
        ]
        y_data = [
            [1.0, 0.0, 0.0],
            [0.3, 0.4, 0.7],
            [0.0, 0.0, 1.0],
            [0.3, 0.0, 0.7],
        ]

        soft_dataset = SoftDataset(
            [
                (x, x_data),
                (y, y_data)
            ],
            weights=weights,
        )

        hard_dataset: HardDataset = expand_soft_dataset(soft_dataset)

        expect: Dict[Instance, float] = {
            (0, 0): sum(_x[0] * _y[0] * _w for _x, _y, _w in zip(x_data, y_data, weights)),
            # (0, 1): sum(_x[0] * _y[1] * _w for _x, _y, _w in zip(x_data, y_data, weights)),  zero weight
            (0, 2): sum(_x[0] * _y[2] * _w for _x, _y, _w in zip(x_data, y_data, weights)),
            (1, 0): sum(_x[1] * _y[0] * _w for _x, _y, _w in zip(x_data, y_data, weights)),
            (1, 1): sum(_x[1] * _y[1] * _w for _x, _y, _w in zip(x_data, y_data, weights)),
            (1, 2): sum(_x[1] * _y[2] * _w for _x, _y, _w in zip(x_data, y_data, weights)),
            (2, 0): sum(_x[2] * _y[0] * _w for _x, _y, _w in zip(x_data, y_data, weights)),
            # (2, 1): sum(_x[2] * _y[1] * _w for _x, _y, _w in zip(x_data, y_data, weights)),  zero weight
            (2, 2): sum(_x[2] * _y[2] * _w for _x, _y, _w in zip(x_data, y_data, weights)),
        }

        self.assertEqual(len(hard_dataset), len(expect))

        instances: Dict[Instance, float] = {
            tuple(
                hard_dataset.state_idxs(rv).item(i) for rv in hard_dataset.rvs
            ): hard_dataset.weights.item(i)
            for i in range(len(hard_dataset))
        }

        self.assertDictEqual(instances, expect)


if __name__ == '__main__':
    test_main()
