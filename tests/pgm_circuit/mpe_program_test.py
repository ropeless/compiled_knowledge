import math
from typing import Optional, Sequence

from ck.pgm import PGM, RandomVariable
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.mpe_program import MPEProgram
from ck.pgm_compiler import factor_elimination
from tests.helpers.unittest_fixture import Fixture, test_main


def compile_pgm(
        pgm: PGM,
        trace_rvs: Optional[Sequence[RandomVariable]] = None,
        const_parameters: bool = True,
) -> MPEProgram:
    pgm_cct: PGMCircuit = factor_elimination.compile_pgm(pgm, const_parameters=const_parameters)
    return MPEProgram(pgm_cct, trace_rvs=trace_rvs, const_parameters=const_parameters)


class Test_MPE(Fixture):
    """
    These are tests for MPEProgram.
    """

    def test_empty(self):
        pgm = PGM()
        mpe_prog = compile_pgm(pgm)

        mpe_result = mpe_prog.mpe()

        self.assertTrue(math.isnan(mpe_result.wmc))
        self.assertEqual(len(mpe_result.mpe), 0)

    def test_one_rv(self):
        pgm = PGM()
        rv = pgm.new_rv('x', 2)
        pgm.new_factor(rv).set_dense().set_flat(0.25, 0.75)

        mpe_prog = compile_pgm(pgm)

        mpe_result = mpe_prog.mpe()

        self.assertEqual(mpe_result.wmc, 0.75)
        self.assertArrayEqual(mpe_result.mpe, [1])

    def test_two_rvs(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)
        pgm.new_factor(x, y).set_dense().set_flat(0.1, 0.4, 0.3, 0.1)

        mpe_prog = compile_pgm(pgm)

        mpe_result = mpe_prog.mpe()

        self.assertEqual(mpe_result.wmc, 0.4)
        self.assertArrayEqual(mpe_result.mpe, [0, 1])

    def test_simple_non_const_params(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)
        pgm.new_factor(x, y).set_dense().set_flat(0.1, 0.4, 0.3, 0.1)

        mpe_prog = compile_pgm(pgm, const_parameters=False)

        mpe_result = mpe_prog.mpe()

        self.assertEqual(mpe_result.wmc, 0.4)
        self.assertArrayEqual(mpe_result.mpe, [0, 1])


if __name__ == '__main__':
    test_main()
