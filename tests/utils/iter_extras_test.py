from typing import List, Iterator

from ck.utils.iter_extras import combos, combos_ranges, flatten, deep_flatten, pairs, sequential_pairs, unzip, powerset, \
    multiply, first
from tests.helpers.unittest_fixture import Fixture, test_main


class TestFlatten(Fixture):

    def test_empty(self):
        it = flatten([])

        self.assertIterFinished(it)

    def test_nested_empty(self):
        it = flatten([[], []])

        self.assertIterFinished(it)

    def test_flat(self):
        it = flatten([[1, 2], [3, 4]])

        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), 3)
        self.assertEqual(next(it), 4)

        self.assertIterFinished(it)

    def test_nested(self):
        it = flatten([[1, [2, 3]], [[4, 5], 6]])

        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), [2, 3])
        self.assertEqual(next(it), [4, 5])
        self.assertEqual(next(it), 6)

        self.assertIterFinished(it)

    def test_double_nested(self):
        it = flatten([[1, [[2, 3], 4]], [[5, [6, 7]], 8]])

        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), [[2, 3], 4])
        self.assertEqual(next(it), [5, [6, 7]])
        self.assertEqual(next(it), 8)

        self.assertIterFinished(it)


class TestDeepFlatten(Fixture):

    def test_empty(self):
        it = deep_flatten([])

        self.assertIterFinished(it)

    def test_nested_empty(self):
        it = deep_flatten([[], []])

        self.assertIterFinished(it)

    def test_flat(self):
        it = deep_flatten([[1, 2], [3, 4]])

        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), 3)
        self.assertEqual(next(it), 4)

        self.assertIterFinished(it)

    def test_nested(self):
        it = deep_flatten([[1, [2, 3]], [[4, 5], 6]])

        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), 3)
        self.assertEqual(next(it), 4)
        self.assertEqual(next(it), 5)
        self.assertEqual(next(it), 6)

        self.assertIterFinished(it)

    def test_double_nested(self):
        it = deep_flatten([[1, [[2, 3], 4]], [[5, [6, 7]], 8]])

        self.assertEqual(next(it), 1)
        self.assertEqual(next(it), 2)
        self.assertEqual(next(it), 3)
        self.assertEqual(next(it), 4)
        self.assertEqual(next(it), 5)
        self.assertEqual(next(it), 6)
        self.assertEqual(next(it), 7)
        self.assertEqual(next(it), 8)

        self.assertIterFinished(it)


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


class TestPairs(Fixture):
    def test_empty(self):
        it = pairs([])

        self.assertIterFinished(it)

    def test_singleton(self):
        it = pairs([1])

        self.assertIterFinished(it)

    def test_2(self):
        it = pairs([1, 2])

        self.assertEqual(next(it), (1, 2))

        self.assertIterFinished(it)

    def test_3(self):
        it = pairs([1, 2, 3])

        self.assertEqual(next(it), (1, 2))
        self.assertEqual(next(it), (1, 3))
        self.assertEqual(next(it), (2, 3))

        self.assertIterFinished(it)

    def test_4(self):
        it = pairs([1, 2, 3, 4])

        self.assertEqual(next(it), (1, 2))
        self.assertEqual(next(it), (1, 3))
        self.assertEqual(next(it), (1, 4))
        self.assertEqual(next(it), (2, 3))
        self.assertEqual(next(it), (2, 4))
        self.assertEqual(next(it), (3, 4))

        self.assertIterFinished(it)


class TestSequentialPairs(Fixture):
    def test_empty(self):
        it = sequential_pairs([])

        self.assertIterFinished(it)

    def test_singleton(self):
        it = sequential_pairs([1])

        self.assertIterFinished(it)

    def test_2(self):
        it = sequential_pairs([1, 2])

        self.assertEqual(next(it), (1, 2))

        self.assertIterFinished(it)

    def test_3(self):
        it = sequential_pairs([1, 2, 3])

        self.assertEqual(next(it), (1, 2))
        self.assertEqual(next(it), (2, 3))

        self.assertIterFinished(it)

    def test_4(self):
        it = sequential_pairs([1, 2, 3, 4])

        self.assertEqual(next(it), (1, 2))
        self.assertEqual(next(it), (2, 3))
        self.assertEqual(next(it), (3, 4))

        self.assertIterFinished(it)


class TestPowerSet(Fixture):
    def test_empty(self):
        it = powerset([])

        self.assertEqual(next(it), ())

        self.assertIterFinished(it)

    def test_singleton(self):
        it = powerset([1])

        self.assertEqual(next(it), ())
        self.assertEqual(next(it), (1,))

        self.assertIterFinished(it)

    def test_2(self):
        it = powerset([1, 2])

        self.assertEqual(next(it), ())
        self.assertEqual(next(it), (1,))
        self.assertEqual(next(it), (2,))
        self.assertEqual(next(it), (1, 2))

        self.assertIterFinished(it)

    def test_3(self):
        it = powerset([1, 2, 3])

        self.assertEqual(next(it), ())
        self.assertEqual(next(it), (1,))
        self.assertEqual(next(it), (2,))
        self.assertEqual(next(it), (3,))
        self.assertEqual(next(it), (1, 2))
        self.assertEqual(next(it), (1, 3))
        self.assertEqual(next(it), (2, 3))
        self.assertEqual(next(it), (1, 2, 3))

        self.assertIterFinished(it)

    def test_4(self):
        it = powerset([1, 2, 3, 4])

        self.assertEqual(next(it), ())
        self.assertEqual(next(it), (1,))
        self.assertEqual(next(it), (2,))
        self.assertEqual(next(it), (3,))
        self.assertEqual(next(it), (4,))
        self.assertEqual(next(it), (1, 2))
        self.assertEqual(next(it), (1, 3))
        self.assertEqual(next(it), (1, 4))
        self.assertEqual(next(it), (2, 3))
        self.assertEqual(next(it), (2, 4))
        self.assertEqual(next(it), (3, 4))
        self.assertEqual(next(it), (1, 2, 3))
        self.assertEqual(next(it), (1, 2, 4))
        self.assertEqual(next(it), (1, 3, 4))
        self.assertEqual(next(it), (2, 3, 4))
        self.assertEqual(next(it), (1, 2, 3, 4))

        self.assertIterFinished(it)


class TestUnzip(Fixture):
    def test_empty(self):
        it = unzip([])

        self.assertIterFinished(it)

    def test_1(self):
        test_input = zip([1, 2, 3])
        it = unzip(test_input)

        self.assertArrayEqual(next(it), [1, 2, 3])

        self.assertIterFinished(it)

    def test_2(self):
        test_input = zip([1, 2, 3], [4, 5, 6])
        it = unzip(test_input)

        self.assertArrayEqual(next(it), [1, 2, 3])
        self.assertArrayEqual(next(it), [4, 5, 6])

        self.assertIterFinished(it)

    def test_3(self):
        test_input = zip([1, 2, 3], [4, 5, 6], [7, 8, 9])
        it = unzip(test_input)

        self.assertArrayEqual(next(it), [1, 2, 3])
        self.assertArrayEqual(next(it), [4, 5, 6])
        self.assertArrayEqual(next(it), [7, 8, 9])

        self.assertIterFinished(it)


class TestMultiply(Fixture):
    def test_empty(self):
        self.assertEqual(multiply([]), 1)

    def test_singleton(self):
        self.assertEqual(multiply([2]), 2)

    def test_2(self):
        self.assertEqual(multiply([2, 3]), 6)

    def test_3(self):
        self.assertEqual(multiply([2, 3, 5]), 30)

    def test_4(self):
        self.assertEqual(multiply([2, 3, 5, 7]), 210)


def iterate(data: List[int]) -> Iterator[int]:
    """
    A generator that yields each element in a list.
    This is used to erase any length information for testing.
    """
    for x in data:
        yield x


class TestFirst(Fixture):

    def test_empty(self):
        with self.assertRaises(Exception):
            first(iterate([]))

    def test_singleton(self):
        self.assertEqual(first(iterate([1])), 1)

    def test_2(self):
        self.assertEqual(first(iterate([2, 1])), 2)

    def test_3(self):
        self.assertEqual(first(iterate([3, 2, 1])), 3)


if __name__ == '__main__':
    test_main()
