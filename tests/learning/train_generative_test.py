from ck.dataset.dataset_from_csv import hard_dataset_from_csv
from ck.learning.parameters import set_potential_functions, set_dense, set_sparse, set_cpt
from ck.learning.train_generative_bn import ParameterValues, get_cpts, train_generative_bn
from ck.pgm import PGM, DensePotentialFunction, SparsePotentialFunction, CPTPotentialFunction
from tests.helpers.unittest_fixture import Fixture, test_main


class TestTrainGenerative(Fixture):

    def test_get_factor_cpts(self):
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

        parameter_values: ParameterValues = get_cpts(pgm, dataset)

        self.assertEqual(parameter_values[0][(0,)], 3 / 9)
        self.assertEqual(parameter_values[0][(1,)], 6 / 9)

        self.assertEqual(parameter_values[1][(0,)], 2 / 9)
        self.assertEqual(parameter_values[1][(1,)], 7 / 9)

        self.assertEqual(parameter_values[2][(0, 0, 0)], 0)
        self.assertEqual(parameter_values[2][(1, 0, 0)], 0)
        self.assertEqual(parameter_values[2][(2, 0, 0)], 1)

        self.assertEqual(parameter_values[2][(0, 0, 1)], 1)
        self.assertEqual(parameter_values[2][(1, 0, 1)], 0)
        self.assertEqual(parameter_values[2][(2, 0, 1)], 0)

        self.assertEqual(parameter_values[2][(0, 1, 0)], 0)
        self.assertEqual(parameter_values[2][(1, 1, 0)], 1 / 2)
        self.assertEqual(parameter_values[2][(2, 1, 0)], 1 / 2)

        self.assertEqual(parameter_values[2][(0, 1, 1)], 2 / 5)
        self.assertEqual(parameter_values[2][(1, 1, 1)], 1 / 5)
        self.assertEqual(parameter_values[2][(2, 1, 1)], 2 / 5)

        self.assertEqual(parameter_values[3][(0, 0)], 1)
        self.assertEqual(parameter_values[3][(1, 0)], 0)

        self.assertEqual(parameter_values[3][(0, 1)], 5 / 7)
        self.assertEqual(parameter_values[3][(1, 1)], 2 / 7)

        self.assertEqual(parameter_values[4][(0, 0)], 2 / 3)
        self.assertEqual(parameter_values[4][(1, 0)], 1 / 3)

        self.assertEqual(parameter_values[4][(0, 1)], 1 / 2)
        self.assertEqual(parameter_values[4][(1, 1)], 1 / 2)

        self.assertEqual(parameter_values[4][(0, 2)], 1 / 4)
        self.assertEqual(parameter_values[4][(1, 2)], 3 / 4)

    def test_get_factor_cpts_with_prior(self):
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

        parameter_values: ParameterValues = get_cpts(pgm, dataset, dirichlet_prior=1)

        self.assertEqual(parameter_values[0][(0,)], (3 + 1) / (9 + 2))
        self.assertEqual(parameter_values[0][(1,)], (6 + 1) / (9 + 2))

        self.assertEqual(parameter_values[1][(0,)], (2 + 1) / (9 + 2))
        self.assertEqual(parameter_values[1][(1,)], (7 + 1) / (9 + 2))

        self.assertEqual(parameter_values[2][(0, 0, 0)], (0 + 1) / (1 + 3))
        self.assertEqual(parameter_values[2][(1, 0, 0)], (0 + 1) / (1 + 3))
        self.assertEqual(parameter_values[2][(2, 0, 0)], (1 + 1) / (1 + 3))

        self.assertEqual(parameter_values[2][(0, 0, 1)], (1 + 1) / (1 + 3))
        self.assertEqual(parameter_values[2][(1, 0, 1)], (0 + 1) / (1 + 3))
        self.assertEqual(parameter_values[2][(2, 0, 1)], (0 + 1) / (1 + 3))

        self.assertEqual(parameter_values[2][(0, 1, 0)], (0 + 1) / (2 + 3))
        self.assertEqual(parameter_values[2][(1, 1, 0)], (1 + 1) / (2 + 3))
        self.assertEqual(parameter_values[2][(2, 1, 0)], (1 + 1) / (2 + 3))

        self.assertEqual(parameter_values[2][(0, 1, 1)], (2 + 1) / (5 + 3))
        self.assertEqual(parameter_values[2][(1, 1, 1)], (1 + 1) / (5 + 3))
        self.assertEqual(parameter_values[2][(2, 1, 1)], (2 + 1) / (5 + 3))

        self.assertEqual(parameter_values[3][(0, 0)], (2 + 1) / (2 + 2))
        self.assertEqual(parameter_values[3][(1, 0)], (0 + 1) / (2 + 2))

        self.assertEqual(parameter_values[3][(0, 1)], (5 + 1) / (7 + 2))
        self.assertEqual(parameter_values[3][(1, 1)], (2 + 1) / (7 + 2))

        self.assertEqual(parameter_values[4][(0, 0)], (2 + 1) / (3 + 2))
        self.assertEqual(parameter_values[4][(1, 0)], (1 + 1) / (3 + 2))

        self.assertEqual(parameter_values[4][(0, 1)], (1 + 1) / (2 + 2))
        self.assertEqual(parameter_values[4][(1, 1)], (1 + 1) / (2 + 2))

        self.assertEqual(parameter_values[4][(0, 2)], (1 + 1) / (4 + 2))
        self.assertEqual(parameter_values[4][(1, 2)], (3 + 1) / (4 + 2))

    def test_train_generative_bn(self):
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

        train_generative_bn(pgm, dataset)

        # These are the expected parameter values, including zeros
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
        self.assertEqual(pgm.factors[2].function[1, 1, 0], 1 / 2)
        self.assertEqual(pgm.factors[2].function[2, 1, 0], 1 / 2)
        self.assertEqual(pgm.factors[2].function[0, 1, 1], 2 / 5)
        self.assertEqual(pgm.factors[2].function[1, 1, 1], 1 / 5)
        self.assertEqual(pgm.factors[2].function[2, 1, 1], 2 / 5)
        self.assertEqual(pgm.factors[3].function[0, 0], 2 / 2)
        self.assertEqual(pgm.factors[3].function[1, 0], 0 / 2)
        self.assertEqual(pgm.factors[3].function[0, 1], 5 / 7)
        self.assertEqual(pgm.factors[3].function[1, 1], 2 / 7)
        self.assertEqual(pgm.factors[4].function[0, 0], 2 / 3)
        self.assertEqual(pgm.factors[4].function[1, 0], 1 / 3)
        self.assertEqual(pgm.factors[4].function[0, 1], 1 / 2)
        self.assertEqual(pgm.factors[4].function[1, 1], 1 / 2)
        self.assertEqual(pgm.factors[4].function[0, 2], 1 / 4)
        self.assertEqual(pgm.factors[4].function[1, 2], 3 / 4)

    def test_set_parameter_values(self):
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

        parameter_values: ParameterValues = get_cpts(pgm, dataset)

        def _assert_param_values():
            # These are the expected parameter values, including zeros
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
            self.assertEqual(pgm.factors[2].function[1, 1, 0], 1 / 2)
            self.assertEqual(pgm.factors[2].function[2, 1, 0], 1 / 2)
            self.assertEqual(pgm.factors[2].function[0, 1, 1], 2 / 5)
            self.assertEqual(pgm.factors[2].function[1, 1, 1], 1 / 5)
            self.assertEqual(pgm.factors[2].function[2, 1, 1], 2 / 5)
            self.assertEqual(pgm.factors[3].function[0, 0], 2 / 2)
            self.assertEqual(pgm.factors[3].function[1, 0], 0 / 2)
            self.assertEqual(pgm.factors[3].function[0, 1], 5 / 7)
            self.assertEqual(pgm.factors[3].function[1, 1], 2 / 7)
            self.assertEqual(pgm.factors[4].function[0, 0], 2 / 3)
            self.assertEqual(pgm.factors[4].function[1, 0], 1 / 3)
            self.assertEqual(pgm.factors[4].function[0, 1], 1 / 2)
            self.assertEqual(pgm.factors[4].function[1, 1], 1 / 2)
            self.assertEqual(pgm.factors[4].function[0, 2], 1 / 4)
            self.assertEqual(pgm.factors[4].function[1, 2], 3 / 4)

        set_potential_functions(pgm, parameter_values)
        for factor in pgm.factors:
            self.assertTrue(isinstance(factor.function, DensePotentialFunction))
        _assert_param_values()

        set_dense(pgm, parameter_values)
        for factor in pgm.factors:
            self.assertTrue(isinstance(factor.function, DensePotentialFunction))
        _assert_param_values()

        set_sparse(pgm, parameter_values)
        for factor in pgm.factors:
            self.assertTrue(isinstance(factor.function, SparsePotentialFunction))
        _assert_param_values()

        set_cpt(pgm, parameter_values)
        for factor in pgm.factors:
            self.assertTrue(isinstance(factor.function, CPTPotentialFunction))
        _assert_param_values()


if __name__ == '__main__':
    test_main()
