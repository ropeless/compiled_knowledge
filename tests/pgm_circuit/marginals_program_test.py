import random
from typing import List

import numpy as np

from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.marginals_program import MarginalsProgram
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import factor_elimination
from ck.utils.np_extras import NDArrayNumeric
from tests.helpers.unittest_fixture import Fixture, test_main


def compile_pgm(pgm: PGM, const_parameters: bool = True) -> MarginalsProgram:
    pgm_cct: PGMCircuit = factor_elimination.compile_pgm(pgm, const_parameters=const_parameters)
    return MarginalsProgram(pgm_cct, const_parameters=const_parameters)


class Test_Marginals(Fixture):
    """
    These are tests for MarginalsProgram.
    """

    def test_marginal_for_rv(self):
        rvs_num_states = [1, 2, 5]

        seed = 123123178243
        random.seed(seed)

        # create a random PGM
        pgm = PGM()
        rvs = [pgm.new_rv(f'x_{i}', num_of_states) for i, num_of_states in enumerate(rvs_num_states)]
        functions = [pgm.new_factor(rv).set_dense().set_stream(random.random).normalise() for rv in rvs]

        marginals = compile_pgm(pgm)

        # test each rv marginal distribution
        for rv, f in zip(rvs, functions):
            dist = marginals.marginal_for_rv(rv)
            expect = np.fromiter((f[j] for j in range(len(rv))), dtype=np.double)
            self.assertEqual(len(dist), len(rv))
            self.assertArrayAlmostEqual(expect, dist)

    def test_result(self):

        rvs_num_states = [1, 2, 5]

        seed = 9817346591
        random.seed(seed)

        # create a random PGM
        pgm = PGM()
        rvs = [pgm.new_rv(f'x_{i}', num_of_states) for i, num_of_states in enumerate(rvs_num_states)]
        functions = [pgm.new_factor(rv).set_dense().set_stream(random.random).normalise() for rv in rvs]

        marginals: MarginalsProgram = compile_pgm(pgm)

        # test
        marginals.compute_conditioned()
        all_marginals: List[NDArrayNumeric] = marginals.result_marginals
        for rv, dist, f in zip(rvs, all_marginals, functions):
            expect = np.fromiter((f[j] for j in range(len(rv))), dtype=np.double)
            self.assertEqual(len(dist), len(rv))
            self.assertArrayAlmostEqual(expect, dist)

    def test_wmc(self):

        rvs_num_states = [1, 2, 5]

        seed = 71598143
        random.seed(seed)

        # create a random PGM
        pgm = PGM()
        rvs = [pgm.new_rv(f'x_{i}', num_of_states) for i, num_of_states in enumerate(rvs_num_states)]
        pgm.new_factor(*rvs).set_dense().set_stream(random.random)

        # compile
        pgm_cct: PGMCircuit = factor_elimination.compile_pgm(pgm)
        wmc = WMCProgram(pgm_cct)
        marginals = MarginalsProgram(pgm_cct)

        # test total wmc
        self.assertEqual(wmc.wmc(), marginals.wmc())
        self.assertEqual(wmc.wmc(), marginals.z)

        # test, conditioning on each indicator
        for indicator in pgm.indicators:
            expect = wmc.wmc(indicator)
            got = marginals.wmc(indicator)
            self.assertEqual(expect, got)

    def test_conditioned_marginals(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)

        f = pgm.new_factor(x, y).set_dense()
        f[(0, 0)] = 0.3
        f[(0, 1)] = 0.7
        f[(1, 0)] = 0.6
        f[(1, 1)] = 0.4

        pgm_cct: PGMCircuit = factor_elimination.compile_pgm(pgm)

        # the test computes conditional probabilities the long way (using wmc) and
        # the short way (using marginals). They should give the same results
        wmc = WMCProgram(pgm_cct)
        marginals = MarginalsProgram(pgm_cct)

        # pr(x)
        wmc_pr_x = np.zeros(2, dtype=np.double)
        wmc_pr_x[0] = wmc.compute_conditioned(x[0]).item()
        wmc_pr_x[1] = wmc.compute_conditioned(x[1]).item()
        wmc_pr_x /= np.sum(wmc_pr_x)

        # pr(x | y=0)
        wmc_pr_x_y0 = np.zeros(2, dtype=np.double)
        wmc_pr_x_y0[0] = wmc.compute_conditioned(x[0], y[0]).item()
        wmc_pr_x_y0[1] = wmc.compute_conditioned(x[1], y[0]).item()
        wmc_pr_x_y0 /= np.sum(wmc_pr_x_y0)

        # pr(x | y=1)
        wmc_pr_x_y1 = np.zeros(2, dtype=np.double)
        wmc_pr_x_y1[0] = wmc.compute_conditioned(x[0], y[1]).item()
        wmc_pr_x_y1[1] = wmc.compute_conditioned(x[1], y[1]).item()
        wmc_pr_x_y1 /= np.sum(wmc_pr_x_y1)

        # pr(y)
        wmc_pr_y = np.zeros(2, dtype=np.double)
        wmc_pr_y[0] = wmc.compute_conditioned(y[0]).item()
        wmc_pr_y[1] = wmc.compute_conditioned(y[1]).item()
        wmc_pr_y /= np.sum(wmc_pr_y)

        # pr(y | x=0)
        wmc_pr_y_x0 = np.zeros(2, dtype=np.double)
        wmc_pr_y_x0[0] = wmc.compute_conditioned(x[0], y[0]).item()
        wmc_pr_y_x0[1] = wmc.compute_conditioned(x[0], y[1]).item()
        wmc_pr_y_x0 /= np.sum(wmc_pr_y_x0)

        # pr(y | x=1)
        wmc_pr_y_x1 = np.zeros(2, dtype=np.double)
        wmc_pr_y_x1[0] = wmc.compute_conditioned(x[1], y[0]).item()
        wmc_pr_y_x1[1] = wmc.compute_conditioned(x[1], y[1]).item()
        wmc_pr_y_x1 /= np.sum(wmc_pr_y_x1)

        marginals.compute_conditioned()
        mar_pr = marginals.result_marginals
        self.assertArrayEqual(wmc_pr_x, mar_pr[0])
        self.assertArrayEqual(wmc_pr_y, mar_pr[1])

        marginals.compute_conditioned(x[0])
        mar_pr = marginals.result_marginals
        self.assertArrayEqual([1, 0], mar_pr[0])
        self.assertArrayEqual(wmc_pr_y_x0, mar_pr[1])

        marginals.compute_conditioned(x[1])
        mar_pr = marginals.result_marginals
        self.assertArrayEqual([0, 1], mar_pr[0])
        self.assertArrayEqual(wmc_pr_y_x1, mar_pr[1])

        marginals.compute_conditioned(y[0])
        mar_pr = marginals.result_marginals
        self.assertArrayEqual(wmc_pr_x_y0, mar_pr[0])
        self.assertArrayEqual([1, 0], mar_pr[1])

        marginals.compute_conditioned(y[1])
        mar_pr = marginals.result_marginals
        self.assertArrayEqual(wmc_pr_x_y1, mar_pr[0])
        self.assertArrayEqual([0, 1], mar_pr[1])

        marginals.compute_conditioned(x[0], y[0])
        mar_pr = marginals.result_marginals
        self.assertArrayEqual([1, 0], mar_pr[0])
        self.assertArrayEqual([1, 0], mar_pr[1])

        marginals.compute_conditioned(x[1], y[0])
        mar_pr = marginals.result_marginals
        self.assertArrayEqual([0, 1], mar_pr[0])
        self.assertArrayEqual([1, 0], mar_pr[1])

        marginals.compute_conditioned(x[0], y[1])
        mar_pr = marginals.result_marginals
        self.assertArrayEqual([1, 0], mar_pr[0])
        self.assertArrayEqual([0, 1], mar_pr[1])

        marginals.compute_conditioned(x[1], y[1])
        mar_pr = marginals.result_marginals
        self.assertArrayEqual([0, 1], mar_pr[0])
        self.assertArrayEqual([0, 1], mar_pr[1])

    def test_simple_non_const_params(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)

        f = pgm.new_factor(x, y).set_dense()
        f[0, 0] = 0.3
        f[0, 1] = 0.7
        f[1, 0] = 0.6
        f[1, 1] = 0.4

        marginals = compile_pgm(pgm, const_parameters=False)

        self.assertAlmostEqual(marginals.z, 2)
        self.assertAlmostEqual(marginals.probability(), 1)
        self.assertAlmostEqual(marginals.probability(x[0]), 0.5)
        self.assertAlmostEqual(marginals.probability(x[1], y[1]), 0.2)


if __name__ == '__main__':
    test_main()
