import random
from typing import Tuple, Sequence, List, Set

from ck.pgm import PGM, Factor, RandomVariable, Instance, Indicator, PotentialFunction
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import factor_elimination
from ck.probability.probability_space import Condition, check_condition
from ck.utils.iter_extras import combos
from tests.helpers.unittest_fixture import Fixture, test_main


def compile_pgm(pgm: PGM, const_parameters: bool = True) -> WMCProgram:
    pgm_cct: PGMCircuit = factor_elimination.compile_pgm(pgm, const_parameters=const_parameters)
    return WMCProgram(pgm_cct, const_parameters=const_parameters)


def compute_map(
        factor: Factor,
        keep_rvs: Sequence[RandomVariable],
        condition: Condition,
) -> Tuple[float, Instance]:
    """
    Compute the MAP the long way, assuming that the given factor
    represents the whole PGM.
    """
    condition: Sequence[Indicator] = check_condition(condition)
    rvs: Sequence[RandomVariable] = factor.rvs
    function: PotentialFunction = factor.function

    inst_map: List[int] = [i for i, rv in enumerate(rvs) if rv in keep_rvs]

    # convert condition to state_matches
    state_matches: List[Set[int]] = [set() for _ in range(len(rvs))]
    rv_index_map = {rv.idx: i for i, rv in enumerate(rvs)}
    for indicator in condition:
        state_matches[rv_index_map[indicator.rv_idx]].add(indicator.state_idx)
    for match, rv in zip(state_matches, rvs):
        if len(match) == 0:
            match.update(rv.state_range())

    def skip(_state_idxs):
        for _state_idx, _match in zip(_state_idxs, state_matches):
            if _state_idx not in _match:
                return True
        return False

    keep_weights = {}
    indicators: Sequence[Indicator]
    for indicators in combos(rvs):
        states = tuple(ind.state_idx for ind in indicators)
        if skip(states):
            continue
        weight = function[states]
        keep_states = tuple(indicators[i].state_idx for i in inst_map)
        keep_weights[keep_states] = keep_weights.get(keep_states, 0) + weight

    best_state = None
    best_weight = float('-inf')
    for states, weight in keep_weights.items():
        if weight > best_weight:
            best_weight = weight
            best_state = states
        elif weight == best_weight:
            raise RuntimeError('testing MAP resulting in a tie')

    return best_weight, best_state


class Test_WMC(Fixture):
    """
    These are tests for WMCProgram.
    Note that there are many related tests in pgm_compiler_test.py.
    """

    def _check_map(self, wmc: WMCProgram, factor, keep_rvs, condition: Condition):
        """Check the MAP calculations of a WMC Program."""
        z = wmc.wmc(*condition)
        best_weight, best_state = compute_map(factor, keep_rvs, condition)
        probability, states = wmc.map(*keep_rvs, condition=condition)
        self.assertAlmostEqual(best_weight / z, probability)
        self.assertEqual(best_state, states)

    def test_empty(self):
        pgm = PGM()
        wmc = compile_pgm(pgm)

        self.assertEqual(1, wmc.wmc())
        self.assertEqual(1, wmc.z)
        self.assertEqual(1, wmc.probability())

    def test_one_rv(self):
        pgm = PGM()
        rv = pgm.new_rv('x', 2)
        f = pgm.new_factor(rv).set_dense().set_stream(random.random)

        total = f[0] + f[1]

        wmc = compile_pgm(pgm)

        self.assertEqual(total, wmc.wmc())
        self.assertEqual(total, wmc.z)
        self.assertEqual(1, wmc.probability())

        self.assertEqual(f[0], wmc.wmc(rv[0]))
        self.assertEqual(f[1], wmc.wmc(rv[1]))

        self.assertEqual(f[0] / total, wmc.probability(rv[0]))
        self.assertEqual(f[1] / total, wmc.probability(rv[1]))

    def test_simple_non_const_params(self):
        pgm = PGM()
        x = pgm.new_rv('x', 2)
        y = pgm.new_rv('y', 2)

        f = pgm.new_factor(x, y).set_dense()
        f[0, 0] = 0.3
        f[0, 1] = 0.7
        f[1, 0] = 0.6
        f[1, 1] = 0.4

        wmc = compile_pgm(pgm, const_parameters=False)

        self.assertAlmostEqual(wmc.z, 2)
        self.assertAlmostEqual(wmc.probability(), 1)
        self.assertAlmostEqual(wmc.probability(x[0]), 0.5)
        self.assertAlmostEqual(wmc.probability(x[1], y[1]), 0.2)

    def test_wmc_map(self):
        number_of_rvs = 4
        num_states_per_rv = 4

        seed = 123123178243
        random.seed(seed)

        # create a random PGM
        pgm = PGM()
        rvs = [pgm.new_rv(f'x_{i}', num_states_per_rv) for i in range(number_of_rvs)]
        factor = pgm.new_factor(*rvs)
        factor.set_dense().set_stream(random.random)

        rv_0 = rvs[0]
        rv_1 = rvs[1]

        # compile
        pgm_cct: PGMCircuit = factor_elimination.compile_pgm(pgm)
        wmc = WMCProgram(pgm_cct)

        # unconditioned, all rvs
        self._check_map(wmc, factor, rvs, ())

        # unconditioned, one rv
        self._check_map(wmc, factor, [rvs[0]], ())

        # simple conditioned, all rvs
        self._check_map(wmc, factor, rvs, (rv_0[0],))

        # simple conditioned, one rv, no overlap
        self._check_map(wmc, factor, [rvs[0]], (rv_1[1],))

        # simple conditioned, one rv, no overlap
        self._check_map(wmc, factor, [rvs[0]], (rv_0[0],))

        # more complex queries

        # unconditioned, some rvs
        self._check_map(wmc, factor, rvs[:2], ())

        # conditioned, all rvs
        self._check_map(wmc, factor, rvs, (rv_0[1], rv_1[0]))

        # conditioned, some rvs, no overlap
        self._check_map(wmc, factor, rvs[2:], (rv_0[1], rv_1[0]))

        # conditioned, some rvs, overlap
        self._check_map(wmc, factor, rvs[:2], (rv_0[1], rv_1[0]))

        # complex conditioned, all rvs
        self._check_map(wmc, factor, rvs, (rv_0[1], rv_0[2], rv_1[0]))

        # complex conditioned, some rvs, no overlap
        self._check_map(wmc, factor, rvs[2:], (rv_0[1], rv_0[2], rv_1[0]))

        # complex conditioned, some rvs, overlap
        self._check_map(wmc, factor, rvs[:2], (rv_0[1], rv_0[2], rv_1[0]))


# class Test_WMC_sampling(Fixture):
#     """
#     These are tests for WMCProgram samplers.
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
#         pgm.new_factor(*rvs).set_dense().set_random()
#
#         # compile
#         wmc = PGM_cct(pgm, freeze_all=True).wmc()
#
#         # get sampler
#         condition_1 = (rvs[1][0],)
#         condition_2 = (rvs[2][1], rvs[2][2])
#         condition = condition_1 + condition_2
#         sampler = sampler_function(wmc, rvs, condition, random)
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
#         pgm.new_factor(*rvs).set_dense().set_random()
#
#         # compile
#         wmc = PGM_cct(pgm, freeze_all=True).wmc()
#
#         # get sampler
#         sampler = sampler_function(wmc, rvs, random)
#
#         self.assertEqual(sampler.rvs, tuple(rvs))
#
#         for _, states in zip(range(num_of_samples), sampler):
#             self.assertEqual(len(states), len(rvs))
#             for rv, state in zip(rvs, states):
#                 self.assertIn(state, rv.state_range())
#
#     def test_direct_conditioned(self):
#         def get_sampler(wmc, rvs, condition, random):
#             return wmc.sample_direct(rvs=rvs, condition=condition, random=random)
#
#         self._test_sampler_with_condition(get_sampler)
#
#     def test_gibbs_conditioned(self):
#         def get_sampler(wmc, rvs, condition, random):
#             return wmc.sample_gibbs(rvs=rvs, condition=condition, random=random)
#
#         self._test_sampler_with_condition(get_sampler)
#
#     def test_metropolis_conditioned(self):
#         def get_sampler(wmc, rvs, condition, random):
#             return wmc.sample_metropolis(rvs=rvs, condition=condition, random=random)
#
#         self._test_sampler_with_condition(get_sampler)
#
#     def test_rejection_conditioned(self):
#         def get_sampler(wmc, rvs, condition, random):
#             return wmc.sample_rejection(rvs=rvs, condition=condition, random=random)
#
#         self._test_sampler_with_condition(get_sampler)
#
#     def test_uniform_conditioned(self):
#         def get_sampler(wmc, rvs, condition, random):
#             return wmc.sample_uniform(rvs=rvs, condition=condition, random=random)
#
#         self._test_sampler_with_condition(get_sampler)
#
#     def test_direct_unconditioned(self):
#         def get_sampler(wmc, rvs, random):
#             return wmc.sample_direct(rvs=rvs, random=random)
#
#         self._test_sampler_without_condition(get_sampler)
#
#     def test_gibbs_unconditioned(self):
#         def get_sampler(wmc, rvs, random):
#             return wmc.sample_gibbs(rvs=rvs, random=random)
#
#         self._test_sampler_without_condition(get_sampler)
#
#     def test_metropolis_unconditioned(self):
#         def get_sampler(wmc, rvs, random):
#             return wmc.sample_metropolis(rvs=rvs, random=random)
#
#         self._test_sampler_without_condition(get_sampler)
#
#     def test_rejection_unconditioned(self):
#         def get_sampler(wmc, rvs, random):
#             return wmc.sample_rejection(rvs=rvs, random=random)
#
#         self._test_sampler_without_condition(get_sampler)
#
#     def test_uniform_unconditioned(self):
#         def get_sampler(wmc, rvs, random):
#             return wmc.sample_uniform(rvs=rvs, random=random)
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
#         pgm.new_factor(*rvs).set_dense().set_random()
#
#         rv_1 = rvs[1]
#         rv_2 = rvs[2]
#         rv_3 = rvs[3]
#
#         # compile
#         wmc = PGM_cct(pgm, freeze_all=True).wmc()
#
#         # construct the sampler - don't change this without also changing the checks.
#         sample_rvs = rvs
#         condition_inds = rv_1[0]
#         markov = [(rv_2, rv_3)]
#
#         sampler = wmc.sample_direct(rvs=sample_rvs, condition=condition_inds, chain_pairs=markov)
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
#         pgm.new_factor(prev_x).set_dense().set_random()
#         pgm.new_factor(x).set_dense().set_random()
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
#         # compile
#         wmc = PGM_cct(pgm, freeze_all=True).wmc()
#
#         sampler = wmc.sample_direct(
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
