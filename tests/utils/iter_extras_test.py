from ck.utils.iter_extras import combos, combos_ranges
from tests.helpers.unittest_fixture import Fixture, test_main


class TestCombos(Fixture):

    def test_combos_empty(self):
        it = combos([])

        self.assertArrayEqual(next(it), [])

        self.assertIterFinished(it)

    def test_combos_single(self):
        it = combos([[1, 2, 3]])

        self.assertArrayEqual(next(it), [1])
        self.assertArrayEqual(next(it), [2])
        self.assertArrayEqual(next(it), [3])

        self.assertIterFinished(it)

    def test_combos_double(self):
        it = combos([[1, 2, 3], [4, 5]])

        self.assertArrayEqual(next(it), [1, 4])
        self.assertArrayEqual(next(it), [2, 4])
        self.assertArrayEqual(next(it), [3, 4])
        self.assertArrayEqual(next(it), [1, 5])
        self.assertArrayEqual(next(it), [2, 5])
        self.assertArrayEqual(next(it), [3, 5])

        self.assertIterFinished(it)


class TestCombosRanges(Fixture):

    def test_combos_ranges_empty(self):
        it = combos_ranges([])

        self.assertArrayEqual(next(it), [])

        self.assertIterFinished(it)

    def test_combos_ranges_single(self):
        it = combos_ranges([3])

        self.assertArrayEqual(next(it), [0])
        self.assertArrayEqual(next(it), [1])
        self.assertArrayEqual(next(it), [2])

        self.assertIterFinished(it)

    def test_combos_ranges_double(self):
        it = combos_ranges([3, 2])

        self.assertArrayEqual(next(it), [0, 0])
        self.assertArrayEqual(next(it), [1, 0])
        self.assertArrayEqual(next(it), [2, 0])
        self.assertArrayEqual(next(it), [0, 1])
        self.assertArrayEqual(next(it), [1, 1])
        self.assertArrayEqual(next(it), [2, 1])

        self.assertIterFinished(it)


if __name__ == '__main__':
    test_main()
