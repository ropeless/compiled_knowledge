from ck.pgm import PGM, RandomVariable
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.target_marginals_program import TargetMarginalsProgram
from ck.pgm_compiler import factor_elimination
from tests.helpers.unittest_fixture import Fixture, test_main


def compile_pgm(pgm: PGM, target_rv: RandomVariable, const_parameters: bool = True) -> TargetMarginalsProgram:
    pgm_cct: PGMCircuit = factor_elimination.compile_pgm(pgm, const_parameters=const_parameters)
    return TargetMarginalsProgram(pgm_cct, target_rv, const_parameters=const_parameters)


class Test_TargetMarginals(Fixture):
    """
    These are tests for TargetMarginalsProgram.
    """

    def test_simple_non_const_params(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)

        f = pgm.new_factor(x, y).set_dense()
        f[0, 0] = 0.3
        f[0, 1] = 0.7
        f[1, 0] = 0.6
        f[1, 1] = 0.4

        tm = compile_pgm(pgm, y, const_parameters=False)

        pr, state_idx = tm.map()
        self.assertAlmostEqual(pr, (0.7 + 0.4) / 2)
        self.assertEqual(state_idx, 1)

        pr, state_idx = tm.map(condition=x[0])
        self.assertAlmostEqual(pr, 0.7)
        self.assertEqual(state_idx, 1)

        pr, state_idx = tm.map(condition=x[1])
        self.assertAlmostEqual(pr, 0.6)
        self.assertEqual(state_idx, 0)

        pr = tm.compute_conditioned(x[0])
        self.assertArrayAlmostEqual([0.3, 0.7], pr)

        pr = tm.compute_conditioned(x[1])
        self.assertArrayAlmostEqual([0.6, 0.4], pr)

    def test_simple_const_params(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)

        f = pgm.new_factor(x, y).set_dense()
        f[(0, 0)] = 0.3
        f[(0, 1)] = 0.7
        f[(1, 0)] = 0.6
        f[(1, 1)] = 0.4

        tm = compile_pgm(pgm, y, const_parameters=True)

        pr, state_idx = tm.map()
        self.assertAlmostEqual(pr, (0.7 + 0.4) / 2)
        self.assertEqual(state_idx, 1)

        pr, state_idx = tm.map(condition=x[0])
        self.assertAlmostEqual(pr, 0.7)
        self.assertEqual(state_idx, 1)

        pr, state_idx = tm.map(condition=x[1])
        self.assertAlmostEqual(pr, 0.6)
        self.assertEqual(state_idx, 0)

        pr = tm.compute_conditioned(x[0])
        self.assertArrayAlmostEqual([0.3, 0.7], pr)

        pr = tm.compute_conditioned(x[1])
        self.assertArrayAlmostEqual([0.6, 0.4], pr)


if __name__ == '__main__':
    test_main()
