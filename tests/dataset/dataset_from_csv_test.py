from io import StringIO
from typing import List

from ck.dataset import HardDataset
from ck.dataset.dataset_from_csv import hard_dataset_from_csv
from ck.pgm import PGM
from ck.utils.tmp_dir import tmp_dir
from tests.helpers.unittest_fixture import Fixture, test_main


class TestHardDataset(Fixture):

    def test_empty(self):
        pgm: PGM = PGM()
        pgm.new_rv('x', 2)
        pgm.new_rv('y', 3)
        pgm.new_rv('z', 4)

        dataset: HardDataset = hard_dataset_from_csv(pgm.rvs, [])

        self.assertEqual(dataset.rvs, pgm.rvs)
        self.assertEqual(len(dataset), 0)

    def test_blank_and_comments(self):
        pgm: PGM = PGM()
        pgm.new_rv('x', 2)
        pgm.new_rv('y', 3)
        pgm.new_rv('z', 4)

        dataset: HardDataset = hard_dataset_from_csv(pgm.rvs, ['', '#', '   ', '   #  comment'])

        self.assertEqual(dataset.rvs, pgm.rvs)
        self.assertEqual(len(dataset), 0)

    def test_no_header(self):
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 3)
        z = pgm.new_rv('z', 4)

        lines: List[str] = [
            '1, 2, 3',
            '0, 1, 2',
        ]
        dataset: HardDataset = hard_dataset_from_csv(pgm.rvs, lines)

        self.assertEqual(dataset.rvs, pgm.rvs)
        self.assertEqual(len(dataset), 2)
        self.assertArrayEqual(dataset.states(x), [1, 0])
        self.assertArrayEqual(dataset.states(y), [2, 1])
        self.assertArrayEqual(dataset.states(z), [3, 2])

    def test_exact_header(self):
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 3)
        z = pgm.new_rv('z', 4)

        lines: List[str] = [
            'x, y, z',
            '1, 2, 3',
            '0, 1, 2',
        ]
        dataset: HardDataset = hard_dataset_from_csv(pgm.rvs, lines)

        self.assertEqual(dataset.rvs, pgm.rvs)
        self.assertEqual(len(dataset), 2)
        self.assertArrayEqual(dataset.states(x), [1, 0])
        self.assertArrayEqual(dataset.states(y), [2, 1])
        self.assertArrayEqual(dataset.states(z), [3, 2])

    def test_useful_header(self):
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 3)
        z = pgm.new_rv('z', 4)

        lines: List[str] = [
            'z, Q, y, x',
            '3, 7, 2, 1',
            '2, 6, 1, 0',
        ]
        dataset: HardDataset = hard_dataset_from_csv(pgm.rvs, lines)

        self.assertEqual(dataset.rvs, pgm.rvs)
        self.assertEqual(len(dataset), 2)
        self.assertArrayEqual(dataset.states(x), [1, 0])
        self.assertArrayEqual(dataset.states(y), [2, 1])
        self.assertArrayEqual(dataset.states(z), [3, 2])

    def test_weights_by_column_number(self):
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 3)
        z = pgm.new_rv('z', 4)

        lines: List[str] = [
            '1, 2.3, 2, 3',
            '0, 4.5, 1, 2',
        ]
        dataset: HardDataset = hard_dataset_from_csv(pgm.rvs, lines, weights=1)

        self.assertEqual(dataset.rvs, pgm.rvs)
        self.assertEqual(len(dataset), 2)
        self.assertArrayEqual(dataset.states(x), [1, 0])
        self.assertArrayEqual(dataset.states(y), [2, 1])
        self.assertArrayEqual(dataset.states(z), [3, 2])
        self.assertArrayEqual(dataset.weights, [2.3, 4.5])

    def test_weights_by_column_negative_number(self):
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 3)
        z = pgm.new_rv('z', 4)

        lines: List[str] = [
            '1, 2, 2.3, 3',
            '0, 1, 4.5, 2',
        ]
        dataset: HardDataset = hard_dataset_from_csv(pgm.rvs, lines, weights=-2)

        self.assertEqual(dataset.rvs, pgm.rvs)
        self.assertEqual(len(dataset), 2)
        self.assertArrayEqual(dataset.states(x), [1, 0])
        self.assertArrayEqual(dataset.states(y), [2, 1])
        self.assertArrayEqual(dataset.states(z), [3, 2])
        self.assertArrayEqual(dataset.weights, [2.3, 4.5])

    def test_weights_by_column_name(self):
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 3)
        z = pgm.new_rv('z', 4)

        lines: List[str] = [
            'z, Q, y, W,   x',
            '3, 7, 2, 2.3, 1',
            '2, 6, 1, 4.5, 0',
        ]
        dataset: HardDataset = hard_dataset_from_csv(pgm.rvs, lines, weights='W')

        self.assertEqual(dataset.rvs, pgm.rvs)
        self.assertEqual(len(dataset), 2)
        self.assertArrayEqual(dataset.states(x), [1, 0])
        self.assertArrayEqual(dataset.states(y), [2, 1])
        self.assertArrayEqual(dataset.states(z), [3, 2])
        self.assertArrayEqual(dataset.weights, [2.3, 4.5])

    def test_missing_column(self):
        pgm: PGM = PGM()
        pgm.new_rv('x', 2)
        pgm.new_rv('y', 3)
        pgm.new_rv('z', 4)

        lines: List[str] = [
            'x, z',
            '1, 3',
            '0, 2',
        ]
        with self.assertRaises(ValueError):
            hard_dataset_from_csv(pgm.rvs, lines)

    def test_inline_comments(self):
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 3)
        z = pgm.new_rv('z', 4)

        lines: List[str] = [
            '1, 2, 3   # this is a comment',
            '0, 1, 2   # and so is this',
        ]
        dataset: HardDataset = hard_dataset_from_csv(pgm.rvs, lines)

        self.assertEqual(dataset.rvs, pgm.rvs)
        self.assertEqual(len(dataset), 2)
        self.assertArrayEqual(dataset.states(x), [1, 0])
        self.assertArrayEqual(dataset.states(y), [2, 1])
        self.assertArrayEqual(dataset.states(z), [3, 2])

    def test_no_comments_fail(self):
        pgm: PGM = PGM()
        pgm.new_rv('x', 2)
        pgm.new_rv('y', 3)
        pgm.new_rv('z', 4)

        lines: List[str] = [
            '1, 2, 3   # this is a comment',
            '0, 1, 2   # and so is this',
        ]

        with self.assertRaises(ValueError):
            hard_dataset_from_csv(pgm.rvs, lines, comment='')

    def test_no_comments_okay(self):
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 3)
        z = pgm.new_rv('z', 4)

        lines: List[str] = [
            '1, 2, 3',
            '0, 1, 2',
        ]
        dataset: HardDataset = hard_dataset_from_csv(pgm.rvs, lines, comment='')

        self.assertEqual(dataset.rvs, pgm.rvs)
        self.assertEqual(len(dataset), 2)
        self.assertArrayEqual(dataset.states(x), [1, 0])
        self.assertArrayEqual(dataset.states(y), [2, 1])
        self.assertArrayEqual(dataset.states(z), [3, 2])

    def test_with_splitlines(self):
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 3)
        z = pgm.new_rv('z', 4)

        data = '''
            1, 2, 3
            0, 1, 2
        '''
        dataset: HardDataset = hard_dataset_from_csv(pgm.rvs, data.splitlines())

        self.assertEqual(dataset.rvs, pgm.rvs)
        self.assertEqual(len(dataset), 2)
        self.assertArrayEqual(dataset.states(x), [1, 0])
        self.assertArrayEqual(dataset.states(y), [2, 1])
        self.assertArrayEqual(dataset.states(z), [3, 2])

    def test_with_string_io(self):
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 3)
        z = pgm.new_rv('z', 4)

        file = StringIO('''
            1, 2, 3
            0, 1, 2
        ''')
        dataset: HardDataset = hard_dataset_from_csv(pgm.rvs, file)

        self.assertEqual(dataset.rvs, pgm.rvs)
        self.assertEqual(len(dataset), 2)
        self.assertArrayEqual(dataset.states(x), [1, 0])
        self.assertArrayEqual(dataset.states(y), [2, 1])
        self.assertArrayEqual(dataset.states(z), [3, 2])

    def test_with_file_io(self):
        pgm: PGM = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 3)
        z = pgm.new_rv('z', 4)

        with tmp_dir():
            with open('data.csv', 'w') as file:
                file.write('''
                    1, 2, 3
                    0, 1, 2
                ''')
            with open('data.csv', 'r') as file:
                dataset: HardDataset = hard_dataset_from_csv(pgm.rvs, file)

        self.assertEqual(dataset.rvs, pgm.rvs)
        self.assertEqual(len(dataset), 2)
        self.assertArrayEqual(dataset.states(x), [1, 0])
        self.assertArrayEqual(dataset.states(y), [2, 1])
        self.assertArrayEqual(dataset.states(z), [3, 2])


if __name__ == '__main__':
    test_main()
