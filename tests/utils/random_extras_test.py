import numpy as np

from ck.utils.random_extras import random_pair, random_permutation
from tests.helpers.unittest_fixture import Fixture, test_main


class TestRandomExtras(Fixture):

    def test_random_pair(self) -> None:
        number_of_trials = 10000
        size = 5

        counts_i = np.zeros((size,))
        counts_j = np.zeros((size,))
        for _ in range(number_of_trials):
            i, j = random_pair(size)
            self.assertTrue(0 <= i < size)
            self.assertTrue(0 <= j < size)
            self.assertTrue(i != j)
            counts_i[i] += 1
            counts_j[j] += 1

        expected_count = number_of_trials / size
        tolerance = expected_count * 0.25  # up to 25% error

        for i in range(size):
            self.assertAlmostEqual(counts_i.item(i), expected_count, delta=tolerance)
            self.assertAlmostEqual(counts_j.item(i), expected_count, delta=tolerance)

    def test_random_permutation(self) -> None:
        number_of_trials = 10000
        size = 5

        counts = np.zeros((size, size))
        for _ in range(number_of_trials):
            perm = random_permutation(size)
            self.assertEqual(len(perm), size)  # correct size
            self.assertEqual(len(set(perm)), size)  # no duplicates
            for i, value in enumerate(perm):
                self.assertTrue(0 <= value < size)
                counts[i][value] += 1

        expected_count = number_of_trials / size
        tolerance = expected_count * 0.25  # up to 25% error

        for i in range(size):
            for j in range(size):
                self.assertAlmostEqual(counts.item(i, j), expected_count, delta=tolerance)


if __name__ == '__main__':
    test_main()
