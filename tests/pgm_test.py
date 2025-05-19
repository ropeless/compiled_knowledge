import pickle
from itertools import count
from unittest import TestCase

from ck.example import Stress
from ck.pgm import PGM, ZeroPotentialFunction, default_pgm_name, Indicator, RVMap
from tests.helpers.unittest_fixture import Fixture, test_main


class PGMFixture(Fixture):

    def assertSameRandomVariables(self, pgm1: PGM, pgm2: PGM):
        """
        Assert that random variables are equal.
        """
        self.assertEqual(pgm1.number_of_rvs, pgm2.number_of_rvs)
        for rv1, rv2 in zip(pgm1.rvs, pgm2.rvs):
            self.assertEqual(rv1.name, rv2.name)
            self.assertEqual(len(rv1), len(rv2))
            for i in range(len(rv1)):
                self.assertEqual(rv1.states[i], rv2.states[i])

    def assertSameFactorsAndFunctions(self, pgm1: PGM, pgm2: PGM):
        """
        Assert that factors are equal.
        """
        self.assertEqual(pgm1.number_of_functions, pgm2.number_of_functions)

        function_idx_1 = {}
        function_idx_2 = {}
        for idx, function1, function2 in zip(count(), pgm1.non_zero_functions, pgm2.non_zero_functions):
            function_idx_1[id(function1)] = idx
            function_idx_2[id(function2)] = idx

            self.assertIs(function1.__class__, function2.__class__)
            self.assertArrayEqual(function1.shape, function1.shape)
            self.assertEqual(function1.number_of_parameters, function2.number_of_parameters)

        def _assertSameFunction(_function1, _function2):
            self.assertIs(_function1.__class__, _function2.__class__)
            if not isinstance(_function1, ZeroPotentialFunction):
                idx1 = function_idx_1[id(_function1)]
                idx2 = function_idx_2[id(_function2)]
                self.assertEqual(idx1, idx2)

        self.assertEqual(pgm1.number_of_factors, pgm2.number_of_factors)
        for factor1, factor2 in zip(pgm1.factors, pgm2.factors):
            self.assertArrayEqual(factor1.shape, factor2.shape)
            _assertSameFunction(factor1.function, factor2.function)

    def assertPGMsFunctionallySame(self, pgm1: PGM, pgm2: PGM, places=6):
        """
        Assert every instance of pgm1 rvs has same value_product as pgm2
        """
        self.assertSameRandomVariables(pgm1, pgm2)

        for inst in pgm1.instances():
            pgm1_value = pgm1.value_product(inst)
            pgm2_value = pgm2.value_product(inst)
            self.assertAlmostEqual(pgm1_value, pgm2_value, places=places)

    def assertPGMsStructureEqual(self, pgm1: PGM, pgm2: PGM):
        """
        Assert that pgm1 and pgm2 have the same random variables, in the same order,
        and that every instantiation has the same WMC (i.e., value_prod).
        """
        self.assertSameRandomVariables(pgm1, pgm2)
        self.assertSameFactorsAndFunctions(pgm1, pgm2)

    def assertEquivalentName(self, pgm1: PGM, pgm2: PGM):
        if pgm1.name == default_pgm_name(pgm1):
            self.assertEqual(pgm2.name, default_pgm_name(pgm2))
        else:
            self.assertEqual(pgm1.name, pgm2.name)


class TestPGM(PGMFixture):

    def test_empty(self):
        pgm = PGM()

        self.assertEqual(0, pgm.number_of_rvs)
        self.assertEqual(0, pgm.number_of_indicators)
        self.assertEqual(1, pgm.number_of_states)
        self.assertEqual(0, pgm.number_of_factors)
        self.assertEqual(0, pgm.number_of_functions)
        self.assertEqual(0, pgm.number_of_non_zero_functions)

    def test_default_name(self):
        pgm = PGM()
        self.assertEqual(pgm.name, default_pgm_name(pgm))

    def test_name(self):
        pgm = PGM('pgm xyz')
        self.assertEqual(pgm.name, 'pgm xyz')

    def test_1_var(self):
        states_per_var = 2

        pgm = PGM()

        a = pgm.new_rv('a', states_per_var)

        self.assertEqual(1, pgm.number_of_rvs)
        self.assertEqual(2, pgm.number_of_indicators)
        self.assertEqual(2, pgm.number_of_states)
        self.assertEqual(0, pgm.number_of_factors)
        self.assertEqual(0, pgm.number_of_functions)
        self.assertEqual(0, pgm.number_of_non_zero_functions)

        self.assertEqual(2, len(a))

        a_factor = pgm.new_factor(a)

        self.assertEqual(1, pgm.number_of_rvs)
        self.assertEqual(2, pgm.number_of_indicators)
        self.assertEqual(2, pgm.number_of_states)
        self.assertEqual(1, pgm.number_of_factors)
        self.assertEqual(1, pgm.number_of_functions)
        self.assertEqual(0, pgm.number_of_non_zero_functions)

        a_factor.set_dense()

        self.assertEqual(1, pgm.number_of_rvs)
        self.assertEqual(2, pgm.number_of_indicators)
        self.assertEqual(2, pgm.number_of_states)
        self.assertEqual(1, pgm.number_of_factors)
        self.assertEqual(1, pgm.number_of_functions)
        self.assertEqual(1, pgm.number_of_non_zero_functions)

    def test_1_arc(self):
        pgm = PGM()

        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 3)

        self.assertEqual(2, pgm.number_of_rvs)
        self.assertEqual(5, pgm.number_of_indicators)
        self.assertEqual(6, pgm.number_of_states)
        self.assertEqual(0, pgm.number_of_factors)
        self.assertEqual(0, pgm.number_of_functions)
        self.assertEqual(0, pgm.number_of_non_zero_functions)

        self.assertEqual(2, len(a))
        self.assertEqual(3, len(b))

        ab_factor = pgm.new_factor(a, b)

        self.assertEqual(2, pgm.number_of_rvs)
        self.assertEqual(5, pgm.number_of_indicators)
        self.assertEqual(6, pgm.number_of_states)
        self.assertEqual(1, pgm.number_of_factors)
        self.assertEqual(1, pgm.number_of_functions)
        self.assertEqual(0, pgm.number_of_non_zero_functions)

        ab_factor.set_dense()

        self.assertEqual(2, pgm.number_of_rvs)
        self.assertEqual(5, pgm.number_of_indicators)
        self.assertEqual(6, pgm.number_of_states)
        self.assertEqual(1, pgm.number_of_factors)
        self.assertEqual(1, pgm.number_of_functions)
        self.assertEqual(1, pgm.number_of_non_zero_functions)

    def test_2_unaries(self):
        pgm = PGM()

        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 3)

        self.assertEqual(2, pgm.number_of_rvs)
        self.assertEqual(5, pgm.number_of_indicators)
        self.assertEqual(6, pgm.number_of_states)
        self.assertEqual(0, pgm.number_of_factors)
        self.assertEqual(0, pgm.number_of_functions)
        self.assertEqual(0, pgm.number_of_non_zero_functions)

        self.assertEqual(2, len(a))
        self.assertEqual(3, len(b))

        a_factor = pgm.new_factor(a)
        b_factor = pgm.new_factor(b)

        self.assertEqual(2, pgm.number_of_rvs)
        self.assertEqual(5, pgm.number_of_indicators)
        self.assertEqual(6, pgm.number_of_states)
        self.assertEqual(2, pgm.number_of_factors)
        self.assertEqual(2, pgm.number_of_functions)
        self.assertEqual(0, pgm.number_of_non_zero_functions)

        a_factor.set_dense()
        b_factor.set_dense()

        self.assertEqual(2, pgm.number_of_rvs)
        self.assertEqual(5, pgm.number_of_indicators)
        self.assertEqual(6, pgm.number_of_states)
        self.assertEqual(2, pgm.number_of_factors)
        self.assertEqual(2, pgm.number_of_functions)
        self.assertEqual(2, pgm.number_of_non_zero_functions)

    def test_2_vars(self):
        pgm = PGM()

        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 3)

        self.assertEqual(2, pgm.number_of_rvs)
        self.assertEqual(5, pgm.number_of_indicators)
        self.assertEqual(6, pgm.number_of_states)
        self.assertEqual(0, pgm.number_of_factors)
        self.assertEqual(0, pgm.number_of_functions)
        self.assertEqual(0, pgm.number_of_non_zero_functions)

        self.assertEqual(2, len(a))
        self.assertEqual(3, len(b))

        a_factor = pgm.new_factor(a)
        b_factor = pgm.new_factor(b)
        ab_factor = pgm.new_factor(a, b)

        self.assertEqual(2, pgm.number_of_rvs)
        self.assertEqual(5, pgm.number_of_indicators)
        self.assertEqual(6, pgm.number_of_states)
        self.assertEqual(3, pgm.number_of_factors)
        self.assertEqual(3, pgm.number_of_functions)
        self.assertEqual(0, pgm.number_of_non_zero_functions)

        a_factor.set_dense()
        b_factor.set_dense()
        ab_factor.set_dense()

        self.assertEqual(2, pgm.number_of_rvs)
        self.assertEqual(5, pgm.number_of_indicators)
        self.assertEqual(6, pgm.number_of_states)
        self.assertEqual(3, pgm.number_of_factors)
        self.assertEqual(3, pgm.number_of_functions)
        self.assertEqual(3, pgm.number_of_non_zero_functions)

    def test_3_vars(self):
        states_per_var = 3

        pgm = PGM()

        a = pgm.new_rv('a', states_per_var)
        b = pgm.new_rv('b', states_per_var)
        c = pgm.new_rv('c', states_per_var)

        self.assertEqual(3, pgm.number_of_rvs)
        self.assertEqual(states_per_var * 3, pgm.number_of_indicators)
        self.assertEqual(states_per_var ** 3, pgm.number_of_states)
        self.assertEqual(0, pgm.number_of_factors)
        self.assertEqual(0, pgm.number_of_functions)
        self.assertEqual(0, pgm.number_of_non_zero_functions)

        self.assertEqual(states_per_var, len(a))
        self.assertEqual(states_per_var, len(b))
        self.assertEqual(states_per_var, len(b))

        ab_factor = pgm.new_factor(a, b)
        c_factor = pgm.new_factor(c)

        self.assertEqual(3, pgm.number_of_rvs)
        self.assertEqual(states_per_var * 3, pgm.number_of_indicators)
        self.assertEqual(states_per_var ** 3, pgm.number_of_states)
        self.assertEqual(2, pgm.number_of_factors)
        self.assertEqual(2, pgm.number_of_functions)
        self.assertEqual(0, pgm.number_of_non_zero_functions)

        ab_factor.set_sparse()
        c_factor.set_sparse()

        self.assertEqual(3, pgm.number_of_rvs)
        self.assertEqual(states_per_var * 3, pgm.number_of_indicators)
        self.assertEqual(states_per_var ** 3, pgm.number_of_states)
        self.assertEqual(2, pgm.number_of_factors)
        self.assertEqual(2, pgm.number_of_functions)
        self.assertEqual(2, pgm.number_of_non_zero_functions)

    def test_higher_order_potentials(self):
        states_per_var = 2

        pgm = PGM()

        a = pgm.new_rv('a', states_per_var)
        b = pgm.new_rv('b', states_per_var)
        c = pgm.new_rv('c', states_per_var)
        d = pgm.new_rv('d', states_per_var)

        pgm.new_factor(a).set_dense()
        pgm.new_factor(a, b).set_dense()
        pgm.new_factor(a, c, d).set_dense()

        self.assertEqual(4, pgm.number_of_rvs)
        self.assertEqual(states_per_var * 4, pgm.number_of_indicators)
        self.assertEqual(states_per_var ** 4, pgm.number_of_states)
        self.assertEqual(3, pgm.number_of_factors)
        self.assertEqual(3, pgm.number_of_functions)
        self.assertEqual(3, pgm.number_of_non_zero_functions)

    def test_loop_3(self):
        states_per_var = 2

        pgm = PGM()

        a = pgm.new_rv('a', states_per_var)
        b = pgm.new_rv('b', states_per_var)
        c = pgm.new_rv('c', states_per_var)

        pgm.new_factor(a, b).set_dense()
        pgm.new_factor(b, c).set_dense()
        pgm.new_factor(c, a).set_dense()

        self.assertEqual(3, pgm.number_of_rvs)
        self.assertEqual(states_per_var * 3, pgm.number_of_indicators)
        self.assertEqual(states_per_var ** 3, pgm.number_of_states)
        self.assertEqual(3, pgm.number_of_factors)
        self.assertEqual(3, pgm.number_of_functions)
        self.assertEqual(3, pgm.number_of_non_zero_functions)

    def test_loop_4(self):
        states_per_var = 2

        pgm = PGM()

        a = pgm.new_rv('a', states_per_var)
        b = pgm.new_rv('b', states_per_var)
        c = pgm.new_rv('c', states_per_var)
        d = pgm.new_rv('d', states_per_var)

        pgm.new_factor(c, a)
        pgm.new_factor(c, b)
        pgm.new_factor(d, c).set_dense()
        pgm.new_factor(b, a).set_sparse()

        self.assertEqual(4, pgm.number_of_rvs)
        self.assertEqual(states_per_var * 4, pgm.number_of_indicators)
        self.assertEqual(states_per_var ** 4, pgm.number_of_states)
        self.assertEqual(4, pgm.number_of_factors)
        self.assertEqual(4, pgm.number_of_functions)
        self.assertEqual(2, pgm.number_of_non_zero_functions)

    def test_instances(self):
        pgm = PGM()

        pgm.new_rv('a', 2)
        pgm.new_rv('b', 3)
        pgm.new_rv('c', 3)

        it = pgm.instances()

        self.assertArrayEqual(next(it), [0, 0, 0])
        self.assertArrayEqual(next(it), [0, 0, 1])
        self.assertArrayEqual(next(it), [0, 0, 2])

        self.assertArrayEqual(next(it), [0, 1, 0])
        self.assertArrayEqual(next(it), [0, 1, 1])
        self.assertArrayEqual(next(it), [0, 1, 2])

        self.assertArrayEqual(next(it), [0, 2, 0])
        self.assertArrayEqual(next(it), [0, 2, 1])
        self.assertArrayEqual(next(it), [0, 2, 2])

        self.assertArrayEqual(next(it), [1, 0, 0])
        self.assertArrayEqual(next(it), [1, 0, 1])
        self.assertArrayEqual(next(it), [1, 0, 2])

        self.assertArrayEqual(next(it), [1, 1, 0])
        self.assertArrayEqual(next(it), [1, 1, 1])
        self.assertArrayEqual(next(it), [1, 1, 2])

        self.assertArrayEqual(next(it), [1, 2, 0])
        self.assertArrayEqual(next(it), [1, 2, 1])
        self.assertArrayEqual(next(it), [1, 2, 2])

        self.assertIterFinished(it)

    def test_value_product(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)

        pgm.new_factor(a, b).set_dense().set_flat(1, 2, 3, 4)
        pgm.new_factor(a).set_dense().set_flat(5, 6)
        pgm.new_factor(b).set_dense().set_flat(7, 8)

        self.assertEqual(pgm.value_product([0, 0]), 1 * 5 * 7)
        self.assertEqual(pgm.value_product([0, 1]), 2 * 5 * 8)
        self.assertEqual(pgm.value_product([1, 0]), 3 * 6 * 7)
        self.assertEqual(pgm.value_product([1, 1]), 4 * 6 * 8)

    def test_value_product_indicators(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)

        pgm.new_factor(a, b).set_dense().set_flat(1, 2, 3, 4)
        pgm.new_factor(a).set_dense().set_flat(5, 6)
        pgm.new_factor(b).set_dense().set_flat(7, 8)

        ab_00 = 1 * 5 * 7
        ab_01 = 2 * 5 * 8
        ab_10 = 3 * 6 * 7
        ab_11 = 4 * 6 * 8

        self.assertEqual(pgm.value_product_indicators(a[0], b[0]), ab_00)
        self.assertEqual(pgm.value_product_indicators(a[0], b[1]), ab_01)
        self.assertEqual(pgm.value_product_indicators(a[1], b[0]), ab_10)
        self.assertEqual(pgm.value_product_indicators(a[1], b[1]), ab_11)

        self.assertEqual(pgm.value_product_indicators(a[0]), ab_00 + ab_01)
        self.assertEqual(pgm.value_product_indicators(a[1]), ab_10 + ab_11)
        self.assertEqual(pgm.value_product_indicators(b[0]), ab_00 + ab_10)
        self.assertEqual(pgm.value_product_indicators(b[1]), ab_01 + ab_11)

        self.assertEqual(pgm.value_product_indicators(), ab_00 + ab_01 + ab_10 + ab_11)

    def test_is_structure_bayesian_empty(self):
        pgm = PGM()
        self.assertEqual(True, pgm.is_structure_bayesian)

    def test_is_structure_bayesian_no_factor(self):
        pgm = PGM()
        pgm.new_rv('A', 2)
        self.assertEqual(False, pgm.is_structure_bayesian)

    def test_is_structure_bayesian_one_var(self):
        pgm = PGM()
        A = pgm.new_rv('A', 2)
        pgm.new_factor(A)
        self.assertEqual(True, pgm.is_structure_bayesian)

    def test_is_structure_bayesian_two_var(self):
        pgm = PGM()
        A = pgm.new_rv('A', 2)
        B = pgm.new_rv('B', 2)
        pgm.new_factor(A, B)
        pgm.new_factor(B)
        self.assertEqual(True, pgm.is_structure_bayesian)

    def test_is_structure_bayesian_two_var_loop(self):
        pgm = PGM()
        A = pgm.new_rv('A', 2)
        B = pgm.new_rv('B', 2)
        pgm.new_factor(A, B)
        pgm.new_factor(B, A)
        self.assertEqual(False, pgm.is_structure_bayesian)

    def test_is_structure_bayesian_three_var(self):
        pgm = PGM()
        A = pgm.new_rv('A', 2)
        B = pgm.new_rv('B', 2)
        C = pgm.new_rv('C', 2)
        pgm.new_factor(A, B)
        pgm.new_factor(C, B)
        pgm.new_factor(B)
        self.assertEqual(True, pgm.is_structure_bayesian)

    def test_is_structure_bayesian_three_loop(self):
        pgm = PGM()
        A = pgm.new_rv('A', 2)
        B = pgm.new_rv('B', 2)
        C = pgm.new_rv('C', 2)
        pgm.new_factor(A, B)
        pgm.new_factor(B, C)
        pgm.new_factor(C, A)
        self.assertEqual(False, pgm.is_structure_bayesian)

    def test_is_structure_bayesian_bad_unary(self):
        pgm = PGM()
        A = pgm.new_rv('A', 2)
        B = pgm.new_rv('B', 2)
        C = pgm.new_rv('C', 2)
        pgm.new_factor(A, B)
        pgm.new_factor(C, B)
        pgm.new_factor(C)
        self.assertEqual(False, pgm.is_structure_bayesian)

    def test_is_structure_bayesian_complex(self):
        pgm = PGM()
        A = pgm.new_rv('A', 2)
        B = pgm.new_rv('B', 2)
        C = pgm.new_rv('C', 2)
        D = pgm.new_rv('D', 2)
        E = pgm.new_rv('E', 2)
        pgm.new_factor(A)
        pgm.new_factor(B)
        pgm.new_factor(C, A, B)
        pgm.new_factor(D, A, B)
        pgm.new_factor(E, C, D)
        self.assertEqual(True, pgm.is_structure_bayesian)

    def test_are_factors_cpts_empty(self):
        pgm = PGM()

        _ = pgm.new_rv('a', 2)
        _ = pgm.new_rv('b', 2)
        _ = pgm.new_rv('c', 2)

        self.assertTrue(pgm.factors_are_cpts())

    def test_are_factors_cpts_zero(self):
        pgm = PGM()

        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        c = pgm.new_rv('c', 2)

        f = pgm.new_factor(a, b, c)

        self.assertTrue(f.function.is_cpt())
        self.assertTrue(pgm.factors_are_cpts())

    def test_are_factors_cpts_dense(self):
        pgm = PGM()

        rain = pgm.new_rv('rain', 2)  # 0=no, 1=yes
        sprinkler = pgm.new_rv('sprinkler', 2)  # 0=sprinkler off, 1=sprinkler on
        grass = pgm.new_rv('grass', 3)  # 0=grass dry, 1=grass damp, 2=grass wet

        factor_g = pgm.new_factor(grass, rain, sprinkler)  # same as a Conditional Probability Table (CPT)
        factor_r = pgm.new_factor(rain)
        factor_s = pgm.new_factor(sprinkler)

        f_g = factor_g.set_dense()
        f_r = factor_r.set_dense()
        f_s = factor_s.set_dense()

        f_r.set_flat(0.8, 0.2)
        f_s.set_flat(0.9, 0.1)
        f_g.set_flat(
            # no,       yes          # rain
            # off, on,  off,  on     # sprinkler
            0.90, 0.01, 0.02, 0.01,  # grass dry
            0.09, 0.01, 0.08, 0.04,  # grass damp
            0.01, 0.98, 0.90, 0.95,  # grass wet
        )

        self.assertTrue(f_g.is_cpt())
        self.assertTrue(f_r.is_cpt())
        self.assertTrue(f_s.is_cpt())

        self.assertTrue(pgm.factors_are_cpts())

        f_g[0, 0, 0] = 99
        self.assertFalse(f_g.is_cpt())
        self.assertFalse(f_g.is_cpt())
        self.assertFalse(pgm.factors_are_cpts())

    def test_are_factors_cpts_sparse(self):
        pgm = PGM()

        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        c = pgm.new_rv('c', 3)

        f = pgm.new_factor(a, b, c)
        pot = f.set_sparse()

        self.assertTrue(f.function.is_cpt())
        self.assertTrue(pgm.factors_are_cpts())

        pot[0, 0, 0] = 0.5
        pot[1, 0, 0] = 0.5
        self.assertTrue(f.function.is_cpt())
        self.assertTrue(pgm.factors_are_cpts())

        pot[0, 0, 1] = 0.4
        pot[1, 0, 1] = 0.6
        self.assertTrue(f.function.is_cpt())
        self.assertTrue(pgm.factors_are_cpts())

        pot[0, 1, 0] = 0.3
        pot[1, 1, 0] = 0.7
        self.assertTrue(f.function.is_cpt())
        self.assertTrue(pgm.factors_are_cpts())

        pot[0, 1, 1] = 0.2
        pot[1, 1, 1] = 0.8
        self.assertTrue(f.function.is_cpt())
        self.assertTrue(pgm.factors_are_cpts())

        pot[0, 0, 0] = 0.0
        self.assertFalse(f.function.is_cpt())
        self.assertFalse(pgm.factors_are_cpts())

        pot[0, 0, 0] = 0.1
        self.assertFalse(f.function.is_cpt())
        self.assertFalse(pgm.factors_are_cpts())

    def test_are_factors_cpts_compact(self):
        pgm = PGM()

        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        c = pgm.new_rv('c', 3)

        f = pgm.new_factor(a, b, c)
        pot = f.set_compact()

        self.assertTrue(f.function.is_cpt())
        self.assertTrue(pgm.factors_are_cpts())

        pot[0, 0, 0] = 0.5
        pot[1, 0, 0] = 0.5
        self.assertTrue(f.function.is_cpt())
        self.assertTrue(pgm.factors_are_cpts())

        pot[0, 0, 1] = 0.4
        pot[1, 0, 1] = 0.6
        self.assertTrue(f.function.is_cpt())
        self.assertTrue(pgm.factors_are_cpts())

        pot[0, 1, 0] = 0.3
        pot[1, 1, 0] = 0.7
        self.assertTrue(f.function.is_cpt())
        self.assertTrue(pgm.factors_are_cpts())

        pot[0, 1, 1] = 0.2
        pot[1, 1, 1] = 0.8
        self.assertTrue(f.function.is_cpt())
        self.assertTrue(pgm.factors_are_cpts())

        pot[0, 0, 0] = 0.0
        self.assertFalse(f.function.is_cpt())
        self.assertFalse(pgm.factors_are_cpts())

        pot[0, 0, 0] = 0.1
        self.assertFalse(f.function.is_cpt())
        self.assertFalse(pgm.factors_are_cpts())

    def test_are_factors_cpts_clause(self):
        pgm = PGM()

        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        c = pgm.new_rv('c', 2)

        f = pgm.new_factor(a, b, c)
        pot = f.set_clause(1, 0, 1)

        self.assertFalse(f.function.is_cpt())
        self.assertFalse(pgm.factors_are_cpts())

        pot.weight = 0
        self.assertTrue(f.function.is_cpt())
        self.assertTrue(pgm.factors_are_cpts())

    def test_sharing_functions(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        c = pgm.new_rv('c', 2)

        f_ab = pgm.new_factor(a, b)
        f_ac = pgm.new_factor(a, c)
        self.assertEqual(pgm.number_of_functions, 2)
        self.assertEqual(pgm.number_of_non_zero_functions, 0)

        f_1 = f_ab.set_dense()
        self.assertEqual(pgm.number_of_functions, 2)
        self.assertEqual(pgm.number_of_non_zero_functions, 1)

        _ = f_ac.set_dense()
        self.assertEqual(pgm.number_of_functions, 2)
        self.assertEqual(pgm.number_of_non_zero_functions, 2)

        _ = f_ac.set_dense()
        self.assertEqual(pgm.number_of_functions, 2)
        self.assertEqual(pgm.number_of_non_zero_functions, 2)

        f_ab.function = f_1
        self.assertEqual(pgm.number_of_functions, 2)
        self.assertEqual(pgm.number_of_non_zero_functions, 2)

        f_ac.function = f_1
        self.assertEqual(pgm.number_of_functions, 1)
        self.assertEqual(pgm.number_of_non_zero_functions, 1)

        f = next(iter(pgm.functions))
        self.assertIs(f_1, f)
        self.assertIs(f_ab.function, f)
        self.assertIs(f_ac.function, f)

        f_ab.set_zero()
        self.assertEqual(pgm.number_of_functions, 2)
        self.assertEqual(pgm.number_of_non_zero_functions, 1)

        f_ac.set_zero()
        self.assertEqual(pgm.number_of_functions, 2)
        self.assertEqual(pgm.number_of_non_zero_functions, 0)

    def test_changing_functions(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        f_ab = pgm.new_factor(a, b)

        p1_ab = f_ab.set_dense()
        p1_ab.set_flat(1, 2, 3, 4)

        self.assertEqual(f_ab.function[0, 0], 1)
        self.assertEqual(f_ab.function[0, 1], 2)
        self.assertEqual(f_ab.function[1, 0], 3)
        self.assertEqual(f_ab.function[1, 1], 4)
        self.assertEqual(pgm.number_of_functions, 1)
        self.assertEqual(pgm.number_of_non_zero_functions, 1)

        z = sum(pgm.value_product(inst) for inst in pgm.instances())
        self.assertEqual(z, 1 + 2 + 3 + 4)

        p2_ab = f_ab.set_dense()
        p2_ab.set_flat(5, 6, 7, 8)

        self.assertEqual(f_ab.function[0, 0], 5)
        self.assertEqual(f_ab.function[0, 1], 6)
        self.assertEqual(f_ab.function[1, 0], 7)
        self.assertEqual(f_ab.function[1, 1], 8)
        self.assertEqual(pgm.number_of_functions, 1)
        self.assertEqual(pgm.number_of_non_zero_functions, 1)

        self.assertEqual(p1_ab[0, 0], 1)
        self.assertEqual(p1_ab[0, 1], 2)
        self.assertEqual(p1_ab[1, 0], 3)
        self.assertEqual(p1_ab[1, 1], 4)

        self.assertEqual(p2_ab[0, 0], 5)
        self.assertEqual(p2_ab[0, 1], 6)
        self.assertEqual(p2_ab[1, 0], 7)
        self.assertEqual(p2_ab[1, 1], 8)

        z = sum(pgm.value_product(inst) for inst in pgm.instances())
        self.assertEqual(z, 5 + 6 + 7 + 8)

    def test_new_factor_implies(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        f_ab = pgm.new_factor_implies(a, 0, b, 1).function

        self.assertEqual(f_ab[0, 0], 0)
        self.assertEqual(f_ab[0, 1], 1)
        self.assertEqual(f_ab[1, 0], 1)
        self.assertEqual(f_ab[1, 1], 1)

    def test_new_factor_equiv(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        f_ab = pgm.new_factor_equiv(a, 0, b, 1).function

        self.assertEqual(f_ab[0, 0], 0)
        self.assertEqual(f_ab[0, 1], 1)
        self.assertEqual(f_ab[1, 0], 1)
        self.assertEqual(f_ab[1, 1], 0)

    def test_new_factor_functional(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 10)

        def mod_2(x):
            return x % 2

        f_ab = pgm.new_factor_functional(mod_2, a, b).function

        for s_b in range(10):
            if mod_2(s_b) == 0:
                expect_0 = 1
                expect_1 = 0
            else:
                expect_0 = 0
                expect_1 = 1
            self.assertEqual(f_ab[0, s_b], expect_0)
            self.assertEqual(f_ab[1, s_b], expect_1)

    def test_number_of_states(self):
        number_of_rvs = 32
        pgm = PGM()

        expect = 1
        for i in range(number_of_rvs):
            num_states = 2 + i
            expect *= num_states
            pgm.new_rv(f'x_{i}', num_states)

        got = pgm.number_of_states
        self.assertEqual(expect, got)

    def test_make_rv_map(self):
        number_of_rvs = 5
        num_states = 2

        pgm = PGM()

        for i in range(number_of_rvs):
            pgm.new_rv(f'x_{i}', num_states)
        rv_map = RVMap(pgm)

        self.assertEqual(len(rv_map), len(pgm.rvs))

        self.assertEqual(rv_map[0], pgm.rvs[0])
        self.assertEqual(rv_map[1], pgm.rvs[1])
        self.assertEqual(rv_map[2], pgm.rvs[2])
        self.assertEqual(rv_map[3], pgm.rvs[3])

        self.assertEqual(rv_map('x_0'), pgm.rvs[0])
        self.assertEqual(rv_map('x_1'), pgm.rvs[1])
        self.assertEqual(rv_map('x_2'), pgm.rvs[2])
        self.assertEqual(rv_map('x_3'), pgm.rvs[3])

        self.assertEqual(rv_map.x_0, pgm.rvs[0])
        self.assertEqual(rv_map.x_1, pgm.rvs[1])
        self.assertEqual(rv_map.x_2, pgm.rvs[2])
        self.assertEqual(rv_map.x_3, pgm.rvs[3])

    def test_indicator_str(self):
        pgm = PGM()
        A = pgm.new_rv('A', 2)
        B = pgm.new_rv('B', ['y', 'n'])

        self.assertEqual('A=0', pgm.indicator_str(A[0]))
        self.assertEqual('A=1', pgm.indicator_str(A[1]))
        self.assertEqual('B=y', pgm.indicator_str(B[0]))
        self.assertEqual('B=n', pgm.indicator_str(B[1]))
        self.assertEqual('A=0, B=n', pgm.indicator_str(A[0], B[1]))

        self.assertEqual('A: 0', pgm.indicator_str(A[0], sep=': '))
        self.assertEqual('A=0; B=n', pgm.indicator_str(A[0], B[1], delim='; '))
        self.assertEqual('A:1; B:y', pgm.indicator_str(A[1], B[0], sep=':', delim='; '))


class TestPGMPickle(PGMFixture):

    def assert_pickle_round_trip(self, pgm: PGM) -> None:
        pkl: bytes = pickle.dumps(pgm)
        clone = pickle.loads(pkl)

        self.assertPGMsStructureEqual(pgm, clone)
        self.assertEqual(pgm.name, clone.name)

    def test_empty(self):
        pgm = PGM()
        self.assert_pickle_round_trip(pgm)

    def test_stress(self):
        pgm = Stress()
        self.assert_pickle_round_trip(pgm)


class TestRandomVariable(PGMFixture):

    def test_pgm(self):
        pgm1 = PGM()
        x = pgm1.new_rv('x', 2)
        pgm2 = PGM()
        y = pgm2.new_rv('y', 2)

        self.assertIs(x.pgm, pgm1)
        self.assertIs(y.pgm, pgm2)
        self.assertIsNot(x.pgm, y.pgm)

    def test_idx(self):
        pgm1 = PGM()
        x = pgm1.new_rv('x', 2)
        y = pgm1.new_rv('y', 2)
        pgm2 = PGM()
        z = pgm2.new_rv('z', 2)

        self.assertEqual(x.idx, 0)
        self.assertEqual(y.idx, 1)
        self.assertEqual(z.idx, 0)

    def test_index(self):
        pgm1 = PGM()
        x = pgm1.new_rv('x', ('zero', 'one', 'two', 'three'))

        pgm2 = PGM()
        y = pgm2.new_rv('y', ('zero', 'one', 'two', 'three', 'four'))

        self.assertIs(x.index(x[0]), 0)
        self.assertIs(x.index(x[1]), 1)
        self.assertIs(x.index(x[2]), 2)
        self.assertIs(x.index(x[3]), 3)

        # These work because Indicators are transferable across PGM instances.
        self.assertIs(x.index(y[0]), 0)
        self.assertIs(x.index(y[1]), 1)
        self.assertIs(x.index(y[2]), 2)
        self.assertIs(x.index(y[3]), 3)

        # Out of range
        y_4: Indicator = y[4]
        with self.assertRaises(ValueError):
            x.index(y_4)

        self.assertIs(x.index(x[1], start=1, stop=3), 1)
        self.assertIs(x.index(x[2], start=1, stop=3), 2)

        # Out of range
        with self.assertRaises(ValueError):
            self.assertIs(x.index(x[0], start=1, stop=3), 0)

        # Out of range
        with self.assertRaises(ValueError):
            self.assertIs(x.index(x[3], start=1, stop=3), 3)

        # Not an indicator
        with self.assertRaises(ValueError):
            x.index('one')

    def test_offset(self):
        pgm = PGM()
        x = pgm.new_rv('x', 3)
        y = pgm.new_rv('y', 5)
        z = pgm.new_rv('z', 7)

        self.assertEqual(x.offset, 0)
        self.assertEqual(y.offset, 3)
        self.assertEqual(z.offset, 8)

    def test_name(self):
        pgm = PGM()
        x = pgm.new_rv('x', 3)
        y = pgm.new_rv(' Y ', 5)
        z = pgm.new_rv(': ! #\n$z$', 7)

        self.assertEqual(x.name, 'x')
        self.assertEqual(y.name, ' Y ')
        self.assertEqual(z.name, ': ! #\n$z$')

    def test_str(self):
        pgm = PGM()
        x = pgm.new_rv('x', 3)
        y = pgm.new_rv(' Y ', 5)
        z = pgm.new_rv(': ! #\n$z$', 7)

        self.assertEqual(str(x), 'x')
        self.assertEqual(str(y), ' Y ')
        self.assertEqual(str(z), ': ! #\n$z$')

    def test_state_index(self):
        pgm = PGM()
        x = pgm.new_rv('x', 3)
        y = pgm.new_rv('y', ['1', '2', '3', None])
        z = pgm.new_rv('z', [None, 'one', 'TWO', 'Three'])

        self.assertEqual(x.state_idx(0), 0)
        self.assertEqual(x.state_idx(1), 1)
        self.assertEqual(x.state_idx(2), 2)

        self.assertEqual(y.state_idx('1'), 0)
        self.assertEqual(y.state_idx('2'), 1)
        self.assertEqual(y.state_idx('3'), 2)
        self.assertEqual(y.state_idx(None), 3)

        self.assertEqual(z.state_idx(None), 0)
        self.assertEqual(z.state_idx('one'), 1)
        self.assertEqual(z.state_idx('TWO'), 2)
        self.assertEqual(z.state_idx('Three'), 3)

    def test_is_default_states(self):
        pgm = PGM()
        x = pgm.new_rv('x', 3)
        y = pgm.new_rv('y', [0, 1, 2])
        z = pgm.new_rv('z', [0, 1, 2, None])

        self.assertTrue(x.is_default_states())
        self.assertTrue(y.is_default_states())
        self.assertFalse(z.is_default_states())

    def test_states(self):
        pgm = PGM()
        x = pgm.new_rv('x', 3)
        y = pgm.new_rv('y', ['1', '2', '3', None])
        z = pgm.new_rv('z', [None, 'one', 'TWO', 'Three'])

        self.assertEqual(x.states[0], 0)
        self.assertEqual(x.states[1], 1)
        self.assertEqual(x.states[2], 2)

        self.assertEqual(y.states[0], '1')
        self.assertEqual(y.states[1], '2')
        self.assertEqual(y.states[2], '3')
        self.assertIsNone(y.states[3])

        self.assertIsNone(z.states[0])
        self.assertEqual(z.states[1], 'one')
        self.assertEqual(z.states[2], 'TWO')
        self.assertEqual(z.states[3], 'Three')

    def test_len(self):
        pgm = PGM()
        x = pgm.new_rv('x', 3)
        y = pgm.new_rv('y', 5)
        z = pgm.new_rv('z', 7)

        self.assertEqual(len(x), 3)
        self.assertEqual(len(y), 5)
        self.assertEqual(len(z), 7)

    def test_indicators(self):
        pgm1 = PGM()
        x = pgm1.new_rv('x', 2)
        y = pgm1.new_rv('y', 2)
        pgm2 = PGM()
        z = pgm2.new_rv('z', 2)

        x_0 = x[0]
        x_1 = x[1]
        y_0 = y[0]
        y_1 = y[1]
        z_0 = z[0]
        z_1 = z[1]

        self.assertEqual(x[0], Indicator(0, 0))
        self.assertEqual(x[1], Indicator(0, 1))
        self.assertEqual(y[0], Indicator(1, 0))
        self.assertEqual(y[1], Indicator(1, 1))
        self.assertEqual(z[0], Indicator(0, 0))
        self.assertEqual(z[1], Indicator(0, 1))

        self.assertEqual(x_0.rv_idx, 0)
        self.assertEqual(x_1.rv_idx, 0)
        self.assertEqual(x_0.state_idx, 0)
        self.assertEqual(x_1.state_idx, 1)

        self.assertEqual(y_0.rv_idx, 1)
        self.assertEqual(y_1.rv_idx, 1)
        self.assertEqual(y_0.state_idx, 0)
        self.assertEqual(y_1.state_idx, 1)

        self.assertNotEqual(x_0, y_0)
        self.assertNotEqual(x_1, y_1)

        self.assertEqual(x_0, z_0)
        self.assertEqual(x_1, z_1)

    def test_indicators_by_state(self):
        pgm = PGM()
        x = pgm.new_rv('x', [1, 2])
        y = pgm.new_rv('y', [3, 4, 5])

        self.assertEqual(x[0], x(1))
        self.assertEqual(x[1], x(2))

        self.assertEqual(y[0], y(3))
        self.assertEqual(y[1], y(4))
        self.assertEqual(y[2], y(5))

    def test_slicing(self):
        pgm1 = PGM()
        x = pgm1.new_rv('x', 6)

        self.assertEqual(x[0], Indicator(0, 0))
        self.assertEqual(x[1], Indicator(0, 1))
        self.assertEqual(x[2], Indicator(0, 2))
        self.assertEqual(x[3], Indicator(0, 3))
        self.assertEqual(x[4], Indicator(0, 4))
        self.assertEqual(x[5], Indicator(0, 5))

        self.assertEqual(x[-6], x[0])
        self.assertEqual(x[-5], x[1])
        self.assertEqual(x[-4], x[2])
        self.assertEqual(x[-3], x[3])
        self.assertEqual(x[-2], x[4])
        self.assertEqual(x[-1], x[5])

        self.assertEqual(x[1:4], (x[1], x[2], x[3]))
        self.assertEqual(x[1:4:2], (x[1], x[3]))
        self.assertEqual(x[1:6:2], (x[1], x[3], x[5]))
        self.assertEqual(x[1::2], (x[1], x[3], x[5]))
        self.assertEqual(x[:5:2], (x[0], x[2], x[4]))
        self.assertEqual(x[::3], (x[0], x[3]))

    def test_state_range(self):
        pgm = PGM()
        x = pgm.new_rv('x', 3)
        y = pgm.new_rv('y', ['1', '2', '3', None])
        z = pgm.new_rv('z', [None, 'one', 'TWO', 'Three'])

        self.assertArrayEqual([0, 1, 2], x.state_range())
        self.assertArrayEqual([0, 1, 2, 3], y.state_range())
        self.assertArrayEqual([0, 1, 2, 3], z.state_range())

    def test_eq(self):
        pgm1 = PGM()
        x = pgm1.new_rv('x', 2)
        y = pgm1.new_rv('y', 2)
        pgm2 = PGM()
        z = pgm2.new_rv('z', 2)

        self.assertEqual(pgm1.rvs[0], x)
        self.assertEqual(pgm1.rvs[1], y)
        self.assertEqual(pgm2.rvs[0], z)

        self.assertNotEqual(x, y)
        self.assertNotEqual(x, z)
        self.assertNotEqual(y, z)

    def test_markov_blanket(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        c = pgm.new_rv('c', 2)
        d = pgm.new_rv('d', 2)
        e = pgm.new_rv('e', 2)

        pgm.new_factor(a, b, c)
        pgm.new_factor(c, d)

        def sort(rvs):
            # Ensure a deterministic order for a set of rvs
            return sorted(rvs, key=lambda rv: rv.name)

        self.assertEqual(sort(a.markov_blanket()), [b, c])
        self.assertEqual(sort(b.markov_blanket()), [a, c])
        self.assertEqual(sort(c.markov_blanket()), [a, b, d])
        self.assertEqual(sort(d.markov_blanket()), [c])
        self.assertEqual(sort(e.markov_blanket()), [])


class TestDensePotentialFunction(PGMFixture):

    def test_DensePotentialFunction(self):
        pgm = PGM()

        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 3)

        f = pgm.new_factor(a, b).set_dense().set_flat(1, 2, 3, 4, 5, 6)

        self.assertEqual(f[(0, 0)], 1)
        self.assertEqual(f[(0, 1)], 2)
        self.assertEqual(f[(0, 2)], 3)
        self.assertEqual(f[(1, 0)], 4)
        self.assertEqual(f[(1, 1)], 5)
        self.assertEqual(f[(1, 2)], 6)

        shape = f.shape
        self.assertEqual(shape, (2, 3))

    def test_DensePotentialFunction_normalise_cpt(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 3)
        f = pgm.new_factor(a, b).set_dense().set_flat(1, 0, 3, 9, 0, 7)

        self.assertEqual(f[(0, 0)], 1)
        self.assertEqual(f[(1, 0)], 9)

        self.assertEqual(f[(0, 1)], 0)
        self.assertEqual(f[(1, 1)], 0)

        self.assertEqual(f[(0, 2)], 3)
        self.assertEqual(f[(1, 2)], 7)

        f.normalise_cpt()

        self.assertEqual(f[(0, 0)], 0.1)
        self.assertEqual(f[(1, 0)], 0.9)

        self.assertEqual(f[(0, 1)], 0.0)
        self.assertEqual(f[(1, 1)], 0.0)

        self.assertEqual(f[(0, 2)], 0.3)
        self.assertEqual(f[(1, 2)], 0.7)

    def test_DensePotentialFunction_set_iter(self):
        pgm = PGM()

        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 3)
        c = pgm.new_rv('c', 2)

        factor_vals_iter = iter([1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16])

        f1 = pgm.new_factor(a, b).set_dense().set_iter(factor_vals_iter)
        f2 = pgm.new_factor(b, c).set_dense().set_iter(factor_vals_iter)

        self.assertEqual(f1[(0, 0)], 1)
        self.assertEqual(f1[(0, 1)], 2)
        self.assertEqual(f1[(0, 2)], 3)
        self.assertEqual(f1[(1, 0)], 4)
        self.assertEqual(f1[(1, 1)], 5)
        self.assertEqual(f1[(1, 2)], 6)

        self.assertEqual(f2[(0, 0)], 11)
        self.assertEqual(f2[(0, 1)], 12)
        self.assertEqual(f2[(1, 0)], 13)
        self.assertEqual(f2[(1, 1)], 14)
        self.assertEqual(f2[(2, 0)], 15)
        self.assertEqual(f2[(2, 1)], 16)

    def test_DensePotentialFunction_set_count(self):
        num_states = 10
        pgm = PGM()
        a = pgm.new_rv('a', num_states)
        f = pgm.new_factor(a).set_dense()

        f.set_iter(count(1))  # 1, 2, 3, ...

        for i in range(num_states):
            self.assertEqual(f[i], i + 1)

    def test_DensePotentialFunction_set_uniform(self):
        num_states = 10
        pgm = PGM()
        a = pgm.new_rv('a', num_states)
        f = pgm.new_factor(a).set_dense().set_uniform()

        for i in range(num_states):
            self.assertEqual(f[i], 1 / num_states)

    def test_DensePotentialFunction_items(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        f = pgm.new_factor(a, b).set_dense()

        f.set_flat(1, 2, 3, 4)
        for param_key, param_value in f.items():
            self.assertEqual(f[param_key], param_value)

    def test_DensePotentialFunction_params(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        f = pgm.new_factor(a, b).set_dense()

        f.set_flat(1, 2, 3, 4)
        for param_key, param_idx, param_value in f.keys_with_param:
            self.assertEqual(f.param_idx(param_key), param_idx)
            self.assertEqual(f.param_value(param_idx), param_value)
            self.assertEqual(f[param_key], param_value)


class TestSparsePotentialFunction(PGMFixture):

    def test_SparsePotentialFunction(self):
        pgm = PGM()

        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 3)

        f = pgm.new_factor(a, b).set_sparse()

        f[0, 0] = 1
        f[0, 2] = 3
        f[1, 1] = 5

        self.assertEqual(f[(0, 0)], 1)
        self.assertIsNone(f.param_idx((0, 1)))
        self.assertEqual(f[(0, 2)], 3)
        self.assertIsNone(f.param_idx((1, 0)))
        self.assertEqual(f[(1, 1)], 5)
        self.assertIsNone(f.param_idx((1, 2)))

        shape = f.shape
        self.assertEqual(shape, (2, 3))

    def test_SparsePotentialFunction_normalise_cpt(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 3)
        f = pgm.new_factor(a, b).set_sparse().set_flat(1, 0, 3, 9, 0, 7)

        self.assertEqual(f[(0, 0)], 1)
        self.assertEqual(f[(1, 0)], 9)

        self.assertEqual(f[(0, 1)], 0)
        self.assertEqual(f[(1, 1)], 0)

        self.assertEqual(f[(0, 2)], 3)
        self.assertEqual(f[(1, 2)], 7)

        self.assertEqual(f.number_of_parameters, 4)

        f.normalise_cpt()

        self.assertEqual(f[(0, 0)], 0.1)
        self.assertEqual(f[(1, 0)], 0.9)

        self.assertEqual(f[(0, 1)], 0.0)
        self.assertEqual(f[(1, 1)], 0.0)

        self.assertEqual(f[(0, 2)], 0.3)
        self.assertEqual(f[(1, 2)], 0.7)

        self.assertEqual(f.number_of_parameters, 4)

    def test_SparsePotentialFunction_set_count(self):
        # A test to exercise a bug
        number_of_states = 10
        pgm = PGM()
        a = pgm.new_rv('a', number_of_states)
        f = pgm.new_factor(a).set_sparse()

        self.assertEqual(f.number_of_parameters, 0)
        self.assertEqual(f.number_of_states, number_of_states)

        f.set_iter(count(1))  # 1, 2, 3, ...

        self.assertEqual(f.number_of_parameters, number_of_states)
        self.assertEqual(f.number_of_states, number_of_states)

        for i in range(number_of_states):
            self.assertEqual(f[i], i + 1)

    def test_SparsePotentialFunction_set_uniform(self):
        num_states = 10
        pgm = PGM()
        a = pgm.new_rv('a', num_states)
        f = pgm.new_factor(a).set_sparse().set_uniform()

        for i in range(num_states):
            self.assertEqual(f[i], 1 / num_states)

    def test_SparsePotentialFunction_parameters_after_delete(self):
        # A test to exercise a bug
        number_of_states = 4
        pgm = PGM()
        a = pgm.new_rv('a', number_of_states)
        f = pgm.new_factor(a).set_sparse()

        f.set_iter(count(1))  # 1, 2, 3, ...
        self.assertIsNotNone(f.param_idx(0))
        self.assertIsNotNone(f.param_idx(1))
        self.assertIsNotNone(f.param_idx(2))
        self.assertIsNotNone(f.param_idx(3))

        f[0] = 0
        self.assertIsNone(f.param_idx(0))
        self.assertIsNotNone(f.param_idx(1))
        self.assertIsNotNone(f.param_idx(2))
        self.assertIsNotNone(f.param_idx(3))

        f[3] = 0
        self.assertIsNone(f.param_idx(0))
        self.assertIsNotNone(f.param_idx(1))
        self.assertIsNotNone(f.param_idx(2))
        self.assertIsNone(f.param_idx(3))

    def test_SparsePotentialFunction_set_iter(self):
        pgm = PGM()

        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 3)
        c = pgm.new_rv('c', 2)

        factor_vals_iter = iter([1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16])

        f1 = pgm.new_factor(a, b).set_sparse().set_iter(factor_vals_iter)
        f2 = pgm.new_factor(b, c).set_sparse().set_iter(factor_vals_iter)

        self.assertEqual(f1[(0, 0)], 1)
        self.assertEqual(f1[(0, 1)], 2)
        self.assertEqual(f1[(0, 2)], 3)
        self.assertEqual(f1[(1, 0)], 4)
        self.assertEqual(f1[(1, 1)], 5)
        self.assertEqual(f1[(1, 2)], 6)

        self.assertEqual(f2[(0, 0)], 11)
        self.assertEqual(f2[(0, 1)], 12)
        self.assertEqual(f2[(1, 0)], 13)
        self.assertEqual(f2[(1, 1)], 14)
        self.assertEqual(f2[(2, 0)], 15)
        self.assertEqual(f2[(2, 1)], 16)

    def test_SparsePotentialFunction_items(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        f = pgm.new_factor(a, b).set_sparse()

        f.set_flat(1, 2, 3, 4)
        for param_key, param_value in f.items():
            self.assertEqual(f[param_key], param_value)

    def test_SparsePotentialFunction_params(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        f = pgm.new_factor(a, b).set_sparse()

        f.set_flat(1, 2, 3, 4)
        for param_key, param_idx, param_value in f.keys_with_param:
            self.assertEqual(f.param_idx(param_key), param_idx)
            self.assertEqual(f.param_value(param_idx), param_value)
            self.assertEqual(f[param_key], param_value)


class TestCompactPotentialFunction(PGMFixture):

    def test_CompactPotentialFunction(self):
        pgm = PGM()

        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 3)

        f = pgm.new_factor(a, b).set_compact()

        f[0, 0] = 1
        f[0, 2] = 1
        f[1, 1] = 2

        self.assertEqual(f[(0, 0)], 1)
        self.assertIsNone(f.param_idx((0, 1)))
        self.assertEqual(f[(0, 2)], 1)
        self.assertIsNone(f.param_idx((1, 0)))
        self.assertEqual(f[(1, 1)], 2)
        self.assertIsNone(f.param_idx((1, 2)))

        shape = f.shape
        self.assertEqual(shape, (2, 3))

        self.assertEqual(f.number_of_parameters, 2)

        p1 = f.param_idx((0, 0))
        p2 = 1 - p1

        self.assertEqual(f.param_value(p1), 1)
        self.assertEqual(f.param_value(p2), 2)

    def test_CompactPotentialFunction_set_count(self):
        number_of_states = 10
        pgm = PGM()
        a = pgm.new_rv('a', number_of_states)
        f = pgm.new_factor(a).set_compact()

        self.assertEqual(f.number_of_parameters, 0)
        self.assertEqual(f.number_of_states, number_of_states)

        f.set_iter(count(1))  # 1, 2, 3, ...

        self.assertEqual(f.number_of_parameters, number_of_states)
        self.assertEqual(f.number_of_states, number_of_states)

        for i in range(number_of_states):
            self.assertEqual(f[i], i + 1)

    def test_CompactPotentialFunction_set_iter(self):
        pgm = PGM()

        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 3)
        c = pgm.new_rv('c', 2)

        factor_vals_iter = iter([1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16])

        f1 = pgm.new_factor(a, b).set_compact().set_iter(factor_vals_iter)
        f2 = pgm.new_factor(b, c).set_compact().set_iter(factor_vals_iter)

        self.assertEqual(f1[(0, 0)], 1)
        self.assertEqual(f1[(0, 1)], 2)
        self.assertEqual(f1[(0, 2)], 3)
        self.assertEqual(f1[(1, 0)], 4)
        self.assertEqual(f1[(1, 1)], 5)
        self.assertEqual(f1[(1, 2)], 6)

        self.assertEqual(f2[(0, 0)], 11)
        self.assertEqual(f2[(0, 1)], 12)
        self.assertEqual(f2[(1, 0)], 13)
        self.assertEqual(f2[(1, 1)], 14)
        self.assertEqual(f2[(2, 0)], 15)
        self.assertEqual(f2[(2, 1)], 16)

    def test_CompactPotentialFunction_set_uniform(self):
        num_states = 10
        pgm = PGM()
        a = pgm.new_rv('a', num_states)
        f = pgm.new_factor(a).set_compact().set_uniform()

        for i in range(num_states):
            self.assertEqual(f[i], 1 / num_states)

    def test_CompactPotentialFunction_parameters_after_delete(self):
        number_of_states = 4
        pgm = PGM()
        a = pgm.new_rv('a', number_of_states)
        f = pgm.new_factor(a).set_compact()

        f.set_iter(count(1))  # 1, 2, 3, ...
        self.assertIsNotNone(f.param_idx(0))
        self.assertIsNotNone(f.param_idx(1))
        self.assertIsNotNone(f.param_idx(2))
        self.assertIsNotNone(f.param_idx(3))

        f[0] = 0
        self.assertIsNone(f.param_idx(0))
        self.assertIsNotNone(f.param_idx(1))
        self.assertIsNotNone(f.param_idx(2))
        self.assertIsNotNone(f.param_idx(3))

        f[3] = 0
        self.assertIsNone(f.param_idx(0))
        self.assertIsNotNone(f.param_idx(1))
        self.assertIsNotNone(f.param_idx(2))
        self.assertIsNone(f.param_idx(3))

    def test_CompactPotentialFunction_items(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        f = pgm.new_factor(a, b).set_compact()

        f.set_flat(1, 2, 3, 4)
        for param_key, param_value in f.items():
            self.assertEqual(f[param_key], param_value)


class TestCPTPotentialFunction(PGMFixture):

    def test_CPTPotentialFunction_set_cpd(self):
        pgm = PGM()
        A = pgm.new_rv('A', 2)
        B = pgm.new_rv('B', 2)
        C = pgm.new_rv('C', 2)

        f_a = pgm.new_factor(A).set_cpt()
        f_a.set_cpd(
            (), [0.4, 0.6]
        )

        f_ba = pgm.new_factor(B, A).set_cpt()
        f_ba.set_cpd(
            (0,), [0.1, 0.9]
        )

        f_cba = pgm.new_factor(C, B, A).set_cpt()
        f_cba.set_cpd(
            (0, 0), [0.1, 0.9]
        ).set_cpd(
            (1, 0), [0.2, 0.8]
        ).set_cpd(
            (0, 1), [0.3, 0.7]
        )

        self.assertEqual(f_a[0], 0.4)
        self.assertEqual(f_a[1], 0.6)

        self.assertEqual(f_ba[0, 0], 0.1)
        self.assertEqual(f_ba[1, 0], 0.9)
        self.assertEqual(f_ba[0, 1], 0.0)
        self.assertEqual(f_ba[1, 1], 0.0)

        self.assertEqual(f_cba[0, 0, 0], 0.1)
        self.assertEqual(f_cba[1, 0, 0], 0.9)
        self.assertEqual(f_cba[0, 1, 0], 0.2)
        self.assertEqual(f_cba[1, 1, 0], 0.8)
        self.assertEqual(f_cba[0, 0, 1], 0.3)
        self.assertEqual(f_cba[1, 0, 1], 0.7)
        self.assertEqual(f_cba[0, 1, 1], 0.0)
        self.assertEqual(f_cba[1, 1, 1], 0.0)

    def test_CPTPotentialFunction_set(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        c = pgm.new_rv('c', 2)
        f = pgm.new_factor(a, b, c).set_cpt()

        self.assertEqual(f[0, 0, 0], 0)
        self.assertEqual(f[0, 0, 1], 0)
        self.assertEqual(f[0, 1, 0], 0)
        self.assertEqual(f[0, 1, 1], 0)
        self.assertEqual(f[1, 0, 0], 0)
        self.assertEqual(f[1, 0, 1], 0)
        self.assertEqual(f[1, 1, 0], 0)
        self.assertEqual(f[1, 1, 1], 0)

        f.set(
            # b  c    a[0] a[1]
            ((0, 0), (0.1, 0.9)),
            ((0, 1), (0.2, 0.8)),
            ((1, 1), (0.4, 0.6)),
        )

        self.assertEqual(f[0, 0, 0], 0.1)
        self.assertEqual(f[0, 0, 1], 0.2)
        self.assertEqual(f[0, 1, 0], 0)
        self.assertEqual(f[0, 1, 1], 0.4)
        self.assertEqual(f[1, 0, 0], 0.9)
        self.assertEqual(f[1, 0, 1], 0.8)
        self.assertEqual(f[1, 1, 0], 0)
        self.assertEqual(f[1, 1, 1], 0.6)

    def test_CPTPotentialFunction_set_None(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        f_a = pgm.new_factor(a).set_cpt()

        self.assertEqual(f_a[0], 0)
        self.assertEqual(f_a[1], 0)

        f_a.set_cpd((), [0.4, 0.6])
        self.assertEqual(f_a[0], 0.4)
        self.assertEqual(f_a[1], 0.6)

        f_a.set_cpd((), None)
        self.assertEqual(f_a[0], 0)
        self.assertEqual(f_a[1], 0)

    def test_CPTPotentialFunction_set_zeros(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        f_a = pgm.new_factor(a).set_cpt()

        self.assertEqual(f_a[0], 0)
        self.assertEqual(f_a[1], 0)

        f_a.set_cpd((), [0.4, 0.6])
        self.assertEqual(f_a[0], 0.4)
        self.assertEqual(f_a[1], 0.6)

        f_a.set_cpd((), [0.0, 0.0])
        self.assertEqual(f_a[0], 0)
        self.assertEqual(f_a[1], 0)

    def test_CPTPotentialFunction_clear(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        f_a = pgm.new_factor(a).set_cpt()

        self.assertEqual(f_a[0], 0)
        self.assertEqual(f_a[1], 0)

        f_a.set_cpd((), [0.4, 0.6])
        self.assertEqual(f_a[0], 0.4)
        self.assertEqual(f_a[1], 0.6)

        f_a.clear_cpd(())
        self.assertEqual(f_a[0], 0)
        self.assertEqual(f_a[1], 0)

    def test_CPTPotentialFunction_set_all(self):
        pgm = PGM()
        A = pgm.new_rv('A', 2)
        B = pgm.new_rv('B', 2)
        C = pgm.new_rv('C', 2)

        f_a = pgm.new_factor(A).set_cpt()
        f_a.set_all(
            [0.4, 0.6]
        )

        f_ba = pgm.new_factor(B, A).set_cpt()
        f_ba.set_all(
            [0.1, 0.9]
        )

        f_cba = pgm.new_factor(C, B, A).set_cpt()
        f_cba.set_all(
            [0.1, 0.9],  # B=0, A=0
            None,  # ..... B=0, A=1
            [0.3, 0.7],  # B=1, A=0
            # missing .... B=1, A=1
        )

        self.assertEqual(f_a[0], 0.4)
        self.assertEqual(f_a[1], 0.6)

        self.assertEqual(f_ba[0, 0], 0.1)
        self.assertEqual(f_ba[1, 0], 0.9)

        self.assertEqual(f_ba[0, 1], 0.0)
        self.assertEqual(f_ba[1, 1], 0.0)

        self.assertEqual(f_cba[0, 0, 0], 0.1)
        self.assertEqual(f_cba[1, 0, 0], 0.9)

        self.assertEqual(f_cba[0, 0, 1], 0.0)
        self.assertEqual(f_cba[1, 0, 1], 0.0)

        self.assertEqual(f_cba[0, 1, 0], 0.3)
        self.assertEqual(f_cba[1, 1, 0], 0.7)

        self.assertEqual(f_cba[0, 1, 1], 0.0)
        self.assertEqual(f_cba[1, 1, 1], 0.0)

    def test_CPTPotentialFunction_set_uniform(self):
        num_states = 10
        pgm = PGM()
        a = pgm.new_rv('a', num_states)
        b = pgm.new_rv('b', num_states)
        c = pgm.new_rv('c', num_states)
        f = pgm.new_factor(a, b, c).set_cpt().set_uniform()

        for i in range(num_states):
            for j in range(num_states):
                for k in range(num_states):
                    self.assertEqual(f[i, j, k], 1 / num_states)

    def test_CPTPotentialFunction_items(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        f = pgm.new_factor(a, b).set_cpt()

        f.set_cpd(0, [0.4, 0.6])
        f.set_cpd(1, [0.3, 0.7])
        for param_key, param_value in f.items():
            self.assertEqual(f[param_key], param_value)

    def test_CPTPotentialFunction_params(self):
        pgm = PGM()
        a = pgm.new_rv('a', 2)
        b = pgm.new_rv('b', 2)
        f = pgm.new_factor(a, b).set_cpt()

        f.set_cpd(0, [0.4, 0.6])
        f.set_cpd(1, [0.3, 0.7])
        for param_key, param_idx, param_value in f.keys_with_param:
            self.assertEqual(f.param_idx(param_key), param_idx)
            self.assertEqual(f.param_value(param_idx), param_value)
            self.assertEqual(f[param_key], param_value)


class TestClausePotentialFunction(PGMFixture):

    def test_ClausePotentialFunction(self):
        pgm = PGM()
        A = pgm.new_rv('A', 2)
        B = pgm.new_rv('B', 2)
        C = pgm.new_rv('C', 2)

        f_abc = pgm.new_factor(A, B, C).set_clause(1, 1, 1)
        clause = f_abc.clause
        self.assertEqual((1, 1, 1), clause)

        f_abc = pgm.new_factor(A, B, C).set_clause(1, 0, 0)
        clause = f_abc.clause
        self.assertEqual((1, 0, 0), clause)

        f_abc = pgm.new_factor(A, B, C).set_clause(1, 0, 1)
        clause = f_abc.clause
        self.assertEqual((1, 0, 1), clause)

        self.assertEqual(f_abc[0, 0, 0], 1)
        self.assertEqual(f_abc[0, 0, 1], 1)
        self.assertEqual(f_abc[0, 1, 0], 0)
        self.assertEqual(f_abc[0, 1, 1], 1)
        self.assertEqual(f_abc[1, 0, 0], 1)
        self.assertEqual(f_abc[1, 0, 1], 1)
        self.assertEqual(f_abc[1, 1, 0], 1)
        self.assertEqual(f_abc[1, 1, 1], 1)

    def test_set_clause(self):
        pgm = PGM()
        A = pgm.new_rv('A', 2)
        B = pgm.new_rv('B', 2)
        C = pgm.new_rv('C', 2)

        f_abc = pgm.new_factor(A, B, C).set_clause(1, 1, 1)
        clause = f_abc.clause
        self.assertEqual((1, 1, 1), clause)

        f_abc.clause = (1, 0, 0)
        clause = f_abc.clause
        self.assertEqual((1, 0, 0), clause)

        f_abc.clause = (1, 0, 1)
        clause = f_abc.clause
        self.assertEqual((1, 0, 1), clause)


class TestRendering(TestCase):

    def test_indicator_str(self):
        pgm = PGM()
        A = pgm.new_rv('A', ['zero', 'one', 'two', 'three'])
        B = pgm.new_rv('B', 2)

        rendered = pgm.indicator_str()
        self.assertEqual(rendered, '')

        rendered = pgm.condition_str(A[1])
        self.assertEqual(rendered, 'A=one')

        rendered = pgm.condition_str(A[2])
        self.assertEqual(rendered, 'A=two')

        rendered = pgm.condition_str(B[0])
        self.assertEqual(rendered, 'B=0')

        rendered = pgm.condition_str(B[1])
        self.assertEqual(rendered, 'B=1')

    def test_instance_str(self):
        pgm = PGM()
        A = pgm.new_rv('A', ['zero', 'one', 'two', 'three'])
        B = pgm.new_rv('B', 2)

        rendered = pgm.instance_str([1, 0])
        self.assertEqual(rendered, 'A=one, B=0')

        rendered = pgm.instance_str([1, 0], rvs=[B, A])
        self.assertEqual(rendered, 'B=1, A=zero')

    def test_condition_str(self):
        pgm = PGM()
        A = pgm.new_rv('A', ['zero', 'one', 'two', 'three'])
        B = pgm.new_rv('B', 2)

        rendered = pgm.condition_str()
        self.assertEqual(rendered, '')

        rendered = pgm.condition_str(B[0])
        self.assertEqual(rendered, 'B=0')

        rendered = pgm.condition_str(B[1])
        self.assertEqual(rendered, 'B=1')

        rendered = pgm.condition_str(A[1], A[2])
        self.assertEqual(rendered, 'A in {one, two}')

        rendered = pgm.condition_str(A[2], A[1])
        self.assertEqual(rendered, 'A in {one, two}')

        rendered = pgm.condition_str(*A)
        self.assertEqual(rendered, 'A in {zero, one, two, three}')

        rendered = pgm.condition_str(A[1], B[0])
        self.assertEqual(rendered, 'A=one, B=0')

        rendered = pgm.condition_str(B[0], A[1])
        self.assertEqual(rendered, 'A=one, B=0')


if __name__ == '__main__':
    test_main()
