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

        self.assertEqual(marginals.z, 2)
        self.assertEqual(marginals.probability(), 1)
        self.assertEqual(marginals.probability(x[0]), 0.5)
        self.assertEqual(marginals.probability(x[1], y[1]), 0.2)


# class Test_Marginals_sampling(Fixture):
#     """
#     These are tests for MarginalsProgram samplers.
#     """
#
#     def _test_sampler_with_condition(self, sampler_function):
#         rv_states = [2, 3, 5, 6, 4]  # number of states for each random variable
#         num_of_samples = 100  # must be at least 2
#
#         seed = 123123178243
#         random.seed(seed)
#
#         # create a random PGM
#         pgm = PGM()
#         rvs = [pgm.new_rv(f'x_{i}', num_states) for i, num_states in enumerate(rv_states)]
#         pgm.new_factor(*rvs).set_dense().set_stream(random.random)
#
#         marginals = compile_pgm(pgm)
#
#         # get sampler
#         condition_1 = (rvs[1][0],)
#         condition_2 = (rvs[2][1], rvs[2][2])
#         condition = condition_1 + condition_2
#         sampler = sampler_function(marginals, rvs, condition, random)
#
#         self.assertEqual(sampler.rvs, tuple(rvs))
#
#         for _, states in zip(range(num_of_samples), sampler):
#             self.assertEqual(len(states), len(rvs))
#             for rv, state in zip(rvs, states):
#                 self.assertIn(state, rv.state_range())
#
#             # condition 1
#             self.assertEqual(states[1], 0)
#
#             # condition 2
#             self.assertIn(states[2], (1, 2))
#
#     def _test_sampler_without_condition(self, sampler_function):
#         rv_states = [2, 3, 5, 6, 4]  # number of states for each random variable
#         num_of_samples = 100  # must be at least 2
#
#         seed = 123123178243
#         random.seed(seed)
#
#         # create a random PGM
#         pgm = PGM()
#         rvs = [pgm.new_rv(f'x_{i}', num_states) for i, num_states in enumerate(rv_states)]
#         pgm.new_factor(*rvs).set_dense().set_stream(random.random)
#
#         marginals = compile_pgm(pgm)
#
#         # get sampler
#         sampler = sampler_function(marginals, rvs, random)
#
#         self.assertEqual(sampler.rvs, tuple(rvs))
#
#         for _, states in zip(range(num_of_samples), sampler):
#             self.assertEqual(len(states), len(rvs))
#             for rv, state in zip(rvs, states):
#                 self.assertIn(state, rv.state_range())
#
#     def test_direct_conditioned(self):
#         def get_sampler(marginals, rvs, condition, random):
#             return marginals.sample_direct(rvs=rvs, condition=condition, random=random)
#
#         self._test_sampler_with_condition(get_sampler)
#
#     def test_direct_unconditioned(self):
#         def get_sampler(marginals, rvs, random):
#             return marginals.sample_direct(rvs=rvs, random=random)
#
#         self._test_sampler_without_condition(get_sampler)
#
#     def test_direct_markov(self):
#         num_of_rvs = 5  # must be at least 4
#         states_per_rv = 3  # must be at least 2
#         num_of_samples = 100  # must be at least 2
#
#         seed = 123123178243
#         random.seed(seed)
#
#         # create a random PGM
#         pgm = PGM()
#         rvs = [pgm.new_rv(f'x_{i}', states_per_rv) for i in range(num_of_rvs)]
#         pgm.new_factor(*rvs).set_dense().set_stream(random.random)
#
#         rv_1 = rvs[1]
#         rv_2 = rvs[2]
#         rv_3 = rvs[3]
#
#         marginals = compile_pgm(pgm)
#
#         # construct the sampler - don't change this without also changing the checks.
#         sample_rvs = rvs
#         condition_inds = rv_1[0]
#         markov = [(rv_2, rv_3)]
#
#         sampler = marginals.sample_direct(rvs=sample_rvs, condition=condition_inds, chain_pairs=markov)
#
#         # generate and check samples
#         prev_states = None
#         for _, states in zip(range(num_of_samples), sampler):
#             self.assertEqual(len(states), len(sample_rvs))
#
#             for rv, state in zip(sample_rvs, states):
#                 self.assertIn(state, rv.state_range())
#
#             # assuming: condition_inds = rv_1[0] & sample_rvs[1] = rv_1
#             self.assertEqual(states[1], 0)
#
#             # assuming: chain_pairs = [(rv_2, rv_3)] & sample_rvs[2] = rv_2 & sample_rvs[3] = rv_3
#             if prev_states is not None:
#                 self.assertEqual(states[3], prev_states[2])
#
#             # prepare for next pass
#             prev_states = states
#
#     def test_direct_markov_deterministic(self):
#         states_per_rv = 4
#         prev_start_state = 0
#         num_of_samples = 10
#
#         pgm = PGM()
#
#         prev_x = pgm.new_rv('prev_x', states_per_rv)
#         x = pgm.new_rv('x', states_per_rv)
#
#         pgm.new_factor(prev_x).set_dense().set_stream(random.random)
#         pgm.new_factor(x).set_dense().set_stream(random.random)
#
#         # this factor will ensure that x = (prev_x + 1) mod states_per_rv
#         f_xx = pgm.new_factor(x, prev_x).set_sparse()
#         for i in range(states_per_rv):
#             j = (i + 1) % states_per_rv
#             f_xx[j, i] = 1
#
#         markov = [(x, prev_x)]  # sample x will be copied to prev_x prior to each sample (except the first sample)
#         initial_markov_condition = (prev_x[prev_start_state])  # prev_x starts in state prev_start_state.
#
#         marginals = compile_pgm(pgm)
#
#         sampler = marginals.sample_direct(
#             rvs=pgm.rvs,
#             condition=(),
#             chain_pairs=markov,
#             initial_chain_condition=initial_markov_condition
#         )
#
#         expect_prev_x = prev_start_state
#         for _, state in zip(range(num_of_samples), sampler):
#             expect_x = (expect_prev_x + 1) % states_per_rv
#             self.assertEqual(state[prev_x.index], expect_prev_x)
#             self.assertEqual(state[x.index], expect_x)
#             expect_prev_x = expect_x


if __name__ == '__main__':
    test_main()
