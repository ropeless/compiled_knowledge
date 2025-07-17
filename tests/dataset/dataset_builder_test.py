import numpy as np
from numpy import nan

from ck.dataset import HardDataset, SoftDataset
from ck.dataset.dataset_builder import DatasetBuilder, Record, hard_dataset_from_builder, soft_dataset_from_builder
from ck.pgm import PGM
from tests.helpers.unittest_fixture import Fixture, test_main


class TestDatasetBuilder(Fixture):

    def test_empty(self):
        builder = DatasetBuilder()
        self.assertEqual(len(builder), 0)
        self.assertEqual(builder.total_weight(), 0)
        self.assertEqual(len(builder.rvs), 0)

        self.assertEqual(len(builder.get_weights()), 0)

    def test_no_records(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        builder = DatasetBuilder([x, y])
        self.assertEqual(len(builder), 0)
        self.assertEqual(builder.total_weight(), 0)
        self.assertEqual(builder.rvs, (x, y))

        self.assertEqual(len(builder.get_weights()), 0)
        self.assertEqual(len(builder.get_column_hard(x)), 0)
        self.assertEqual(len(builder.get_column_soft(x)), 0)

    def test_simple(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        builder = DatasetBuilder([x, y])
        self.assertEqual(len(builder), 0)
        self.assertEqual(builder.total_weight(), 0)
        self.assertEqual(builder.rvs, (x, y))

        record: Record = builder.append()
        self.assertEqual(len(builder), 1)
        self.assertEqual(builder.total_weight(), 1)
        self.assertEqual(len(record), 2)
        self.assertIsNone(record[0])
        self.assertIsNone(record[1])

        record: Record = builder.append(1, 2)
        self.assertEqual(len(builder), 2)
        self.assertEqual(builder.total_weight(), 2)
        self.assertEqual(len(record), 2)
        self.assertEqual(record[0], 1)
        self.assertEqual(record[1], 2)

        record: Record = builder.append(None, [0.7, 0.1, 0.2])
        self.assertEqual(len(builder), 3)
        self.assertEqual(builder.total_weight(), 3)
        self.assertEqual(len(record), 2)
        self.assertIsNone(record[0])
        self.assertArrayEqual(record[1], [0.7, 0.1, 0.2])

        weights = builder.get_weights()
        self.assertEqual(weights.shape, (3,))
        self.assertArrayEqual(weights, [1, 1, 1])

        builder[1].weight = 3
        self.assertEqual(len(builder), 3)
        self.assertEqual(builder.total_weight(), 5)
        weights = builder.get_weights()
        self.assertEqual(weights.shape, (3,))
        self.assertArrayEqual(weights, [1, 3, 1])

        column = builder.get_column_hard(x, missing=99)
        self.assertEqual(column.shape, (3,))
        self.assertArrayEqual(column, [99, 1, 99])

        column = builder.get_column_hard(y)
        self.assertEqual(column.shape, (3,))
        self.assertArrayEqual(column, [3, 2, 0])

        column = builder.get_column_soft(x)
        self.assertNDArrayEqual(column, np.array([[nan, nan], [0, 1], [nan, nan]]), nan_equality=True)

        column = builder.get_column_soft(y, missing=[-1, -1, -1])
        self.assertNDArrayEqual(column, np.array([[-1, -1, -1], [0, 0, 1], [0.7, 0.1, 0.2]]))

    def test_insert_and_del(self):
        pgm = PGM()
        x = pgm.new_rv('x', 10)

        builder = DatasetBuilder([x])
        builder.append(0)
        builder.append(1)
        builder.append(2)

        self.assertArrayEqual(builder.get_column_hard(x), [0, 1, 2])

        builder.insert(0, [3])
        self.assertArrayEqual(builder.get_column_hard(x), [3, 0, 1, 2])

        builder.insert(-1, [4])
        self.assertArrayEqual(builder.get_column_hard(x), [3, 0, 1, 4, 2])

        builder.insert(len(builder), [5])
        self.assertArrayEqual(builder.get_column_hard(x), [3, 0, 1, 4, 2, 5])

        del builder[1]
        self.assertArrayEqual(builder.get_column_hard(x), [3, 1, 4, 2, 5])

        del builder[-1]
        self.assertArrayEqual(builder.get_column_hard(x), [3, 1, 4, 2])

    def test_get_item(self):
        pgm = PGM()
        x = pgm.new_rv('x', 10)

        builder = DatasetBuilder([x])
        builder.append(1)
        builder.append(3)
        builder.append(5)
        builder.append(2)
        builder.append(7)
        builder.append(4)
        self.assertArrayEqual(builder.get_column_hard(x), [1, 3, 5, 2, 7, 4])

        record = builder[1]
        self.assertEqual(record[0], 3)

        record = builder[-1]
        self.assertEqual(record[0], 4)

        records = builder[2:4]
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0][0], 5)
        self.assertEqual(records[1][0], 2)

        with self.assertRaises(IndexError):
            _ = builder[6]

        with self.assertRaises(IndexError):
            _ = builder[-7]

    def test_ensure_column(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))
        z = pgm.new_rv('z', 4)

        builder = DatasetBuilder([x, y])
        builder.ensure_column(y, z)
        self.assertEqual(builder.rvs, (x, y, z))

    def test_del_column(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        builder = DatasetBuilder([x, y])
        self.assertEqual(builder.rvs, (x, y))

        builder.append(1, 2)
        builder.append(0, 1)

        builder.del_column(x)
        self.assertEqual(builder.rvs, (y,))
        self.assertEqual(builder[0][0], 2)
        self.assertEqual(builder[1][0], 1)

    def test_record_get_item(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))
        z = pgm.new_rv('z', 4)

        builder = DatasetBuilder([x, y, z])
        record = builder.append(1, 2, 3)

        self.assertEqual(record[0], 1)
        self.assertEqual(record[-1], 3)
        self.assertEqual(record[x], 1)
        self.assertEqual(record[y], 2)
        self.assertArrayEqual(record[1:3], [2, 3])

        with self.assertRaises(IndexError):
            _ = record[3]

        with self.assertRaises(IndexError):
            _ = record[-4]

    def test_record_bad_set(self):
        pgm = PGM()
        pgm.new_rv('x', (True, False))
        pgm.new_rv('y', ('yes', 'no', 'maybe'))

        builder = DatasetBuilder(pgm.rvs)
        record = builder.append()

        with self.assertRaises(ValueError):
            record.set(0, 0, 0)  # too many values

        with self.assertRaises(ValueError):
            record.set(0)  # not enough values

    def test_record_set_states(self):
        pgm = PGM()
        pgm.new_rv('x', (True, False))
        pgm.new_rv('y', ('yes', 'no', 'maybe'))

        builder = DatasetBuilder(pgm.rvs)
        record = builder.append()

        record.set_states(True, 'yes')
        self.assertArrayEqual(record, [0, 0])

        record.set_states(False, 'maybe')
        self.assertArrayEqual(record, [1, 2])

        with self.assertRaises(Exception):
            record.set_states(True, True)

    def test_record_set_item(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))
        z = pgm.new_rv('z', 4)

        builder = DatasetBuilder([x, y, z])
        record = builder.append()

        record[0] = 1
        self.assertEqual(record[0], 1)

        record[-1] = 3
        self.assertEqual(record[2], 3)

        record[y] = 0
        self.assertEqual(record[1], 0)

        record[0] = (0.9, 0.1)

        self.assertArrayEqual(record[0], (0.9, 0.1))

        record[1:3] = [2, 1]
        self.assertArrayEqual(record[1:3], [2, 1])

        record[1:3] = [[0.5, 0.5, 0.0], 2]
        self.assertArrayEqual(record[1], [0.5, 0.5, 0.0])
        self.assertEqual(record[2], 2)

        record[0] = None
        self.assertIsNone(record[0])

        with self.assertRaises(IndexError):
            record[3] = 0

        with self.assertRaises(IndexError):
            record[-4] = 0

        with self.assertRaises(ValueError):
            record[0] = (0.0, 1.0, 0.0)  # too many state weights

        with self.assertRaises(ValueError):
            record[2] = (0.0, 1.0, 0.0)  # no enough state weights

        with self.assertRaises(ValueError):
            record[0] = 2  # state index out of range

        with self.assertRaises(ValueError):
            record[0] = -1  # state index negative

    def test_append_dataset_hard(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))
        z = pgm.new_rv('z', ('1', '2'))

        dataset = HardDataset([
            (x, [0, 0, 1, 1, 0]),
            (y, [1, 1, 1, 0, 0]),
            (z, [1, 1, 0, 1, 1]),
        ])

        builder = DatasetBuilder([x, y, z])
        builder.append_dataset(dataset)

        self.assertEqual(len(builder), 5)
        self.assertArrayEqual(builder[0], [0, 1, 1])
        self.assertArrayEqual(builder[1], [0, 1, 1])
        self.assertArrayEqual(builder[2], [1, 1, 0])
        self.assertArrayEqual(builder[3], [1, 0, 1])
        self.assertArrayEqual(builder[4], [0, 0, 1])

    def test_append_dataset_soft(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))
        z = pgm.new_rv('z', ('1', '2'))

        dataset = SoftDataset([
            (x, np.array([[0.6, 0.4], [1.0, 0.0]])),
            (y, np.array([[0.6, 0.3, 0.1], [0.0, 0.0, 1.0]])),
            (z, np.array([[0.6, 0.4], [1.0, 0.0]])),
        ])

        builder = DatasetBuilder([x, y, z])
        builder.append_dataset(dataset)

        self.assertEqual(len(builder), 2)
        self.assertArrayEqual(builder[0], [[0.6, 0.4], [0.6, 0.3, 0.1], [0.6, 0.4]])
        self.assertArrayEqual(builder[1], [[1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0]])

    def test_hard_dataset_from_builder(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        builder = DatasetBuilder([x, y])
        builder.append()
        builder.append(1, 2).weight = 3
        builder.append(None, [0.7, 0.1, 0.2])

        dataset: HardDataset = hard_dataset_from_builder(builder, missing=99)
        self.assertEqual(dataset.rvs, (x, y))
        self.assertEqual(len(dataset), 3)
        self.assertArrayEqual(dataset.state_idxs(x), [99, 1, 99])
        self.assertArrayEqual(dataset.state_idxs(y), [99, 2, 0])

    def test_soft_dataset_from_builder(self):
        pgm = PGM()
        x = pgm.new_rv('x', (True, False))
        y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

        builder = DatasetBuilder([x, y])
        builder.append()
        builder.append(1, 2).weight = 3
        builder.append(None, [0.7, 0.1, 0.2])

        dataset: SoftDataset = soft_dataset_from_builder(builder, missing=-1)
        self.assertEqual(dataset.rvs, (x, y))
        self.assertEqual(len(dataset), 3)
        self.assertNDArrayEqual(dataset.state_weights(x), np.array([[-1, -1], [0, 1], [-1, -1]]))
        self.assertNDArrayEqual(dataset.state_weights(y), np.array([[-1, -1, -1], [0, 0, 1], [0.7, 0.1, 0.2]]))


if __name__ == '__main__':
    test_main()
