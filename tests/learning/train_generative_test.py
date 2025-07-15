from ck.dataset.dataset_from_csv import hard_dataset_from_csv
from ck.learning.train_generative import ParameterValues, train_generative_bn
from ck.pgm import PGM, ZeroPotentialFunction, DensePotentialFunction, SparsePotentialFunction, CPTPotentialFunction
from tests.helpers.unittest_fixture import Fixture, test_main


class TestTrainGenerative(Fixture):

    def test_train_generative(self):
        pgm = PGM('Student')
        difficult = pgm.new_rv('difficult', ['y', 'n'])
        intelligent = pgm.new_rv('intelligent', ['y', 'n'])
        grade = pgm.new_rv('grade', ['low', 'medium', 'high'])
        award = pgm.new_rv('award', ['y', 'n'])
        letter = pgm.new_rv('letter', ['y', 'n'])
        pgm.new_factor(difficult)
        pgm.new_factor(intelligent)
        pgm.new_factor(grade, intelligent, difficult)
        pgm.new_factor(award, intelligent)
        pgm.new_factor(letter, grade)

        rvs = (difficult, intelligent, grade, award, letter)
        csv = """
        0,1,2,0,1
        1,1,2,0,1
        1,1,2,0,1
        0,0,2,0,0
        0,1,1,1,0
        1,1,1,1,1
        1,1,0,0,0
        1,1,0,0,1
        1,0,0,0,0
        """
        dataset = hard_dataset_from_csv(rvs, csv.splitlines())

        parameter_values: ParameterValues = train_generative_bn(pgm, dataset)

        self.assertIs(parameter_values.pgm, pgm)
        self.assertArrayEqual(parameter_values.cpts[0][()], [3 / 9, 6 / 9])
        self.assertArrayEqual(parameter_values.cpts[1][()], [2 / 9, 7 / 9])
        self.assertArrayEqual(parameter_values.cpts[2][(0, 0)], [0, 0, 1])
        self.assertArrayEqual(parameter_values.cpts[2][(0, 1)], [1, 0, 0])
        self.assertArrayEqual(parameter_values.cpts[2][(1, 0)], [0, 1 / 2, 1 / 2])
        self.assertArrayEqual(parameter_values.cpts[2][(1, 1)], [2 / 5, 1 / 5, 2 / 5])
        self.assertArrayEqual(parameter_values.cpts[3][(0,)], [2 / 2, 0])
        self.assertArrayEqual(parameter_values.cpts[3][(1,)], [5 / 7, 2 / 7])
        self.assertArrayEqual(parameter_values.cpts[4][(0,)], [2/3, 1/3])
        self.assertArrayEqual(parameter_values.cpts[4][(1,)], [1/2, 1/2])
        self.assertArrayEqual(parameter_values.cpts[4][(2,)], [1/4, 3/4])

    def test_parameter_values(self):
        pgm = PGM('Student')
        difficult = pgm.new_rv('difficult', ['y', 'n'])
        intelligent = pgm.new_rv('intelligent', ['y', 'n'])
        grade = pgm.new_rv('grade', ['low', 'medium', 'high'])
        award = pgm.new_rv('award', ['y', 'n'])
        letter = pgm.new_rv('letter', ['y', 'n'])
        pgm.new_factor(difficult)
        pgm.new_factor(intelligent)
        pgm.new_factor(grade, intelligent, difficult)
        pgm.new_factor(award, intelligent)
        pgm.new_factor(letter, grade)

        rvs = (difficult, intelligent, grade, award, letter)
        csv = """
        0,1,2,0,1
        1,1,2,0,1
        1,1,2,0,1
        0,0,2,0,0
        0,1,1,1,0
        1,1,1,1,1
        1,1,0,0,0
        1,1,0,0,1
        1,0,0,0,0
        """
        dataset = hard_dataset_from_csv(rvs, csv.splitlines())

        parameter_values: ParameterValues = train_generative_bn(pgm, dataset)

        parameter_values.set_zero()
        for factor in pgm.factors:
            self.assertTrue(isinstance(factor.function, ZeroPotentialFunction))

        parameter_values.set_dense()
        for factor in pgm.factors:
            self.assertTrue(isinstance(factor.function, DensePotentialFunction))
        self.assertEqual(pgm.factors[0].function[0], 3 / 9)
        self.assertEqual(pgm.factors[0].function[1], 6 / 9)
        self.assertEqual(pgm.factors[1].function[0], 2 / 9)
        self.assertEqual(pgm.factors[1].function[1], 7 / 9)
        self.assertEqual(pgm.factors[2].function[0, 0, 0], 0)
        self.assertEqual(pgm.factors[2].function[1, 0, 0], 0)
        self.assertEqual(pgm.factors[2].function[2, 0, 0], 1)
        self.assertEqual(pgm.factors[2].function[0, 0, 1], 1)
        self.assertEqual(pgm.factors[2].function[1, 0, 1], 0)
        self.assertEqual(pgm.factors[2].function[2, 0, 1], 0)
        self.assertEqual(pgm.factors[2].function[0, 1, 0], 0)
        self.assertEqual(pgm.factors[2].function[1, 1, 0], 1/2)
        self.assertEqual(pgm.factors[2].function[2, 1, 0], 1/2)
        self.assertEqual(pgm.factors[2].function[0, 1, 1], 2/5)
        self.assertEqual(pgm.factors[2].function[1, 1, 1], 1/5)
        self.assertEqual(pgm.factors[2].function[2, 1, 1], 2/5)
        self.assertEqual(pgm.factors[3].function[0, 0], 2/2)
        self.assertEqual(pgm.factors[3].function[1, 0], 0/2)
        self.assertEqual(pgm.factors[3].function[0, 1], 5/7)
        self.assertEqual(pgm.factors[3].function[1, 1], 2/7)
        self.assertEqual(pgm.factors[4].function[0, 0], 2/3)
        self.assertEqual(pgm.factors[4].function[1, 0], 1/3)
        self.assertEqual(pgm.factors[4].function[0, 1], 1/2)
        self.assertEqual(pgm.factors[4].function[1, 1], 1/2)
        self.assertEqual(pgm.factors[4].function[0, 2], 1/4)
        self.assertEqual(pgm.factors[4].function[1, 2], 3/4)

        parameter_values.set_sparse()
        for factor in pgm.factors:
            self.assertTrue(isinstance(factor.function, SparsePotentialFunction))
        self.assertEqual(pgm.factors[0].function[0], 3 / 9)
        self.assertEqual(pgm.factors[0].function[1], 6 / 9)
        self.assertEqual(pgm.factors[1].function[0], 2 / 9)
        self.assertEqual(pgm.factors[1].function[1], 7 / 9)
        self.assertEqual(pgm.factors[2].function[0, 0, 0], 0)
        self.assertEqual(pgm.factors[2].function[1, 0, 0], 0)
        self.assertEqual(pgm.factors[2].function[2, 0, 0], 1)
        self.assertEqual(pgm.factors[2].function[0, 0, 1], 1)
        self.assertEqual(pgm.factors[2].function[1, 0, 1], 0)
        self.assertEqual(pgm.factors[2].function[2, 0, 1], 0)
        self.assertEqual(pgm.factors[2].function[0, 1, 0], 0)
        self.assertEqual(pgm.factors[2].function[1, 1, 0], 1/2)
        self.assertEqual(pgm.factors[2].function[2, 1, 0], 1/2)
        self.assertEqual(pgm.factors[2].function[0, 1, 1], 2/5)
        self.assertEqual(pgm.factors[2].function[1, 1, 1], 1/5)
        self.assertEqual(pgm.factors[2].function[2, 1, 1], 2/5)
        self.assertEqual(pgm.factors[3].function[0, 0], 2/2)
        self.assertEqual(pgm.factors[3].function[1, 0], 0/2)
        self.assertEqual(pgm.factors[3].function[0, 1], 5/7)
        self.assertEqual(pgm.factors[3].function[1, 1], 2/7)
        self.assertEqual(pgm.factors[4].function[0, 0], 2/3)
        self.assertEqual(pgm.factors[4].function[1, 0], 1/3)
        self.assertEqual(pgm.factors[4].function[0, 1], 1/2)
        self.assertEqual(pgm.factors[4].function[1, 1], 1/2)
        self.assertEqual(pgm.factors[4].function[0, 2], 1/4)
        self.assertEqual(pgm.factors[4].function[1, 2], 3/4)

        parameter_values.set_cpt()
        for factor in pgm.factors:
            self.assertTrue(isinstance(factor.function, CPTPotentialFunction))
        self.assertEqual(pgm.factors[0].function[0], 3 / 9)
        self.assertEqual(pgm.factors[0].function[1], 6 / 9)
        self.assertEqual(pgm.factors[1].function[0], 2 / 9)
        self.assertEqual(pgm.factors[1].function[1], 7 / 9)
        self.assertEqual(pgm.factors[2].function[0, 0, 0], 0)
        self.assertEqual(pgm.factors[2].function[1, 0, 0], 0)
        self.assertEqual(pgm.factors[2].function[2, 0, 0], 1)
        self.assertEqual(pgm.factors[2].function[0, 0, 1], 1)
        self.assertEqual(pgm.factors[2].function[1, 0, 1], 0)
        self.assertEqual(pgm.factors[2].function[2, 0, 1], 0)
        self.assertEqual(pgm.factors[2].function[0, 1, 0], 0)
        self.assertEqual(pgm.factors[2].function[1, 1, 0], 1/2)
        self.assertEqual(pgm.factors[2].function[2, 1, 0], 1/2)
        self.assertEqual(pgm.factors[2].function[0, 1, 1], 2/5)
        self.assertEqual(pgm.factors[2].function[1, 1, 1], 1/5)
        self.assertEqual(pgm.factors[2].function[2, 1, 1], 2/5)
        self.assertEqual(pgm.factors[3].function[0, 0], 2/2)
        self.assertEqual(pgm.factors[3].function[1, 0], 0/2)
        self.assertEqual(pgm.factors[3].function[0, 1], 5/7)
        self.assertEqual(pgm.factors[3].function[1, 1], 2/7)
        self.assertEqual(pgm.factors[4].function[0, 0], 2/3)
        self.assertEqual(pgm.factors[4].function[1, 0], 1/3)
        self.assertEqual(pgm.factors[4].function[0, 1], 1/2)
        self.assertEqual(pgm.factors[4].function[1, 1], 1/2)
        self.assertEqual(pgm.factors[4].function[0, 2], 1/4)
        self.assertEqual(pgm.factors[4].function[1, 2], 3/4)


if __name__ == '__main__':
    test_main()
