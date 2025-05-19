from unittest import TestCase, main as test_main

from ck.pgm import PGM
from ck.probability.empirical_probability_space import EmpiricalProbabilitySpace


class TestEmpiricalProbabilitySpace(TestCase):

    def test_no_rv(self):
        pr = EmpiricalProbabilitySpace((), ())

        self.assertEqual(pr.z, 0)
        self.assertEqual(pr.wmc(), 0)

    def test_one_rv(self):
        x = PGM().new_rv('x', ['a', 'b', 'c', 'd', 'e', 'f'])
        samples = [0, 1, 1, 1, 1, 2, 2, 3, 4, 4]

        pr = EmpiricalProbabilitySpace(
            (x,),
            ([sample] for sample in samples)
        )

        self.assertEqual(pr.z, 10)
        self.assertEqual(pr.wmc(), 10)

        self.assertEqual(pr.wmc(x[0]), 1)
        self.assertEqual(pr.wmc(x[1]), 4)
        self.assertEqual(pr.wmc(x[2]), 2)
        self.assertEqual(pr.wmc(x[3]), 1)
        self.assertEqual(pr.wmc(x[4]), 2)
        self.assertEqual(pr.wmc(x[5]), 0)

        # x is 1 or 4
        self.assertEqual(pr.wmc(x[1], x[4]), 6)

    def test_two_rvs(self):
        pgm = PGM()
        x = pgm.new_rv('x', ['a', 'b', 'c', 'd', 'e', 'f'])
        y = pgm.new_rv('y', ['1', '2', '3', '4', '5', '6'])
        samples = [
            [0, 0],
            [1, 0],
            [1, 1],
            [1, 1],
            [1, 2],
            [2, 2],
            [2, 2],
            [3, 2],
            [4, 4],
            [4, 5],
        ]

        pr = EmpiricalProbabilitySpace((x, y), samples)

        self.assertEqual(pr.z, 10)
        self.assertEqual(pr.wmc(), 10)

        self.assertEqual(pr.wmc(x[0]), 1)
        self.assertEqual(pr.wmc(x[1]), 4)
        self.assertEqual(pr.wmc(x[2]), 2)
        self.assertEqual(pr.wmc(x[3]), 1)
        self.assertEqual(pr.wmc(x[4]), 2)
        self.assertEqual(pr.wmc(x[5]), 0)

        # x is 1 or 4
        self.assertEqual(pr.wmc(x[1], x[4]), 6)

        self.assertEqual(pr.wmc(y[0]), 2)
        self.assertEqual(pr.wmc(y[1]), 2)
        self.assertEqual(pr.wmc(y[2]), 4)
        self.assertEqual(pr.wmc(y[3]), 0)
        self.assertEqual(pr.wmc(y[4]), 1)
        self.assertEqual(pr.wmc(y[5]), 1)

        # y is 1 or 4
        self.assertEqual(pr.wmc(y[1], y[4]), 3)

        # x is 1 and y is 2
        self.assertEqual(pr.wmc(x[1], y[2]), 1)

        # x is 2 and y is 2
        self.assertEqual(pr.wmc(x[2], y[2]), 2)

        # x is 3 and y is 2
        self.assertEqual(pr.wmc(x[3], y[2]), 1)

        # x is 1 or 2, and, y is 1 or 2
        self.assertEqual(pr.wmc(x[1], x[2], y[1], y[2]), 5)


if __name__ == '__main__':
    test_main()
