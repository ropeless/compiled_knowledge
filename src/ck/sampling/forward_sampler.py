"""
This module defines a Sampler for Bayesian networks using
the Forward Sampling method.
"""
import random
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple, Sequence, Dict, List, Set

import numpy as np

from ck.pgm import PGM, RandomVariable, PotentialFunction, Instance, Indicator
from ck.probability.probability_space import Condition, check_condition, dtype_for_state_indexes
from ck.sampling.sampler import Sampler
from ck.sampling.sampler_support import YieldF
from ck.utils.np_extras import NDArrayStates, DTypeStates
from ck.utils.random_extras import Random


class ForwardSampler(Sampler):
    """
    A ForwardSampler operates directly on a Bayesian network PGM.
    It does not compile the PGM but directly uses the CPTs.

    It determines the parent random variables to sample (recursively)
    and samples them in a 'forward' order.

    Conditioning is implemented by rejecting samples incompatible with the condition,
    thus if the probability of the condition is low, the sampler will reject
    many samples, which may be slow.

    When unconditioned, this sampler can be very efficient.
    """

    def __init__(
            self,
            pgm: PGM,
            rvs: Optional[RandomVariable | Sequence[RandomVariable]] = None,
            condition: Condition = (),
            rand: Random = random,
            check_is_bayesian_network: bool = True
    ):
        """
        Construct a forward sampler for the given PGM.

        Args:
            pgm: is the model to be sampled. It is assumed to be a proper Bayesian network.
            rvs: the list of random variables to sample; the yielded state vectors are
                co-indexed with rvs; if None, then the WMC rvs are used; if rvs is a single
                random variable, then single samples are yielded.
            condition: is a collection of zero or more conditioning indicators.
            rand: provides the stream of random numbers.
            check_is_bayesian_network: is a Boolean flag. If true, then the
                constructor will raise an exception if pgm.check_is_bayesian_network()
                is false.
        """
        if check_is_bayesian_network and not pgm.check_is_bayesian_network():
            raise ValueError('the given PGM is not a Bayesian network')

        condition: Sequence[Indicator] = check_condition(condition)

        self._yield_f: YieldF
        if rvs is None:
            # all input rvs
            rvs = pgm.rvs
            self._yield_f = lambda x: x.tolist()
        elif rvs in pgm.rvs:
            # a single rv
            rvs = (rvs,)
            self._yield_f = lambda x: x.item()
        else:
            # a sequence of rvs
            self._yield_f = lambda x: x.tolist()

        super().__init__(rvs, condition)

        # Create a map from rv_idx to its factor.
        # This assumes a Bayesian network structure - one factor per rv.
        rv_factors = {
            factor.rvs[0].idx: factor
            for factor in pgm.factors
        }

        # Infer sampling order of random variables.
        # Infer the mapping from random variables to yielded samples.
        # Get a _SampleRV for each rv in rvs, and for pre-requisites.
        # This assumes a Bayesian network structure - no directed loops.
        sample_rvs_map: Dict[int, _SampleRV] = {}  # map from rv index to _SampleRV object
        sample_rvs_list: List[_SampleRV] = []  # sequence of _SampleRV objects for sampling order
        output_sample_rvs: Sequence[_SampleRV] = tuple(
            _get_sample_rv(rv.idx, rv_factors, sample_rvs_map, sample_rvs_list)
            for rv in rvs
        )

        # Add constraints to sample rvs, based on conditions.
        ind: Indicator
        for ind in condition:
            sample_rv: _SampleRV = _get_sample_rv(ind.rv_idx, rv_factors, sample_rvs_map, sample_rvs_list)
            constraints = sample_rv.constraints
            if constraints is None:
                constraints = sample_rv.constraints = set()
            constraints.add(ind.state_idx)

        # Store needed data.
        self._dtype: DTypeStates = dtype_for_state_indexes(pgm.rvs)
        self._rand: Random = rand
        self._sample_rvs_list: Sequence[_SampleRV] = sample_rvs_list
        self._output_sample_rvs: Sequence[_SampleRV] = output_sample_rvs

    def __iter__(self) -> Iterator[Instance] | Iterator[int]:
        yield_f = self._yield_f
        output_sample_rvs = self._output_sample_rvs

        # Allocate working memory for yielded result
        instance: NDArrayStates = np.zeros(len(output_sample_rvs), dtype=self._dtype)

        # Allocate working memory for sampled states
        states: NDArrayStates = np.zeros(len(self._sample_rvs_list), dtype=self._dtype)

        while True:
            success = False
            while not success:
                success = self._set_sample(states)

            for i, sample_rv in enumerate(output_sample_rvs):
                instance[i] = states[sample_rv.val_idx]

            yield yield_f(instance)

    def _set_sample(self, states: NDArrayStates) -> bool:
        """
        Set the states array with a sample.
        The states array is co-indexed with self._sample_rvs_list.
        If the sample has zero probability (i.e. it should be rejected)
        then False is returned, otherwise True is returned.
        """
        sample_rvs_list = self._sample_rvs_list
        rand = self._rand

        for sample_rv in sample_rvs_list:
            state = sample_rv.sample(rand, states)

            # Check that a sample was possible (i.e., positive conditioned
            # probability for some state of `sample_rv`).
            if state is None:
                return False

            # Check for possible rejection based on conditioning
            if sample_rv.constraints is not None and state not in sample_rv.constraints:
                return False

            states[sample_rv.val_idx] = state
        return True


@dataclass
class _SampleRV:
    rv: RandomVariable
    function: PotentialFunction
    val_idx: int
    parent_val_indexes: Sequence[int]
    constraints: Optional[Set[int]]

    def sample(self, rand: Random, states: NDArrayStates) -> Optional[int]:
        """
        Return a random state index for the random variable, using
        the appropriate distribution over possible states,
        conditioned on the parent states that can be found in 'states'.

        The function returns None if there was no state with a non-zero probability.

        Returns:
            state index from this random variable, or None if no state with a non-zero probability.
        """
        num_states: int = len(self.rv)
        function: PotentialFunction = self.function
        parent_val_indexes: Sequence[int] = self.parent_val_indexes

        x: float = rand.random()  # uniform variate in [0, 1)

        # Get the cumulative CPD, conditioned on parent states.
        parent_states: Tuple[int, ...] = tuple(states.item(i) for i in parent_val_indexes)
        total: float = 0
        for state in range(num_states):
            total += function[(state,) + parent_states]
            if x < total:
                return state

        if total <= 0:
            # No state with positive probability
            return None
        else:
            return num_states - 1


def _get_sample_rv(rv_idx: int, rv_factors, sample_rvs_map, sample_rvs_list) -> _SampleRV:
    """
    Get a _SampleRV for the indexed random variable.
    This may be called recursively for the parents of the nominated rv.
    """
    sample_rv = sample_rvs_map.get(rv_idx)
    if sample_rv is None:
        factor = rv_factors[rv_idx]
        rv = factor.rvs[0]
        parent_val_indexes = tuple(
            _get_sample_rv(parent.idx, rv_factors, sample_rvs_map, sample_rvs_list).val_idx
            for parent in factor[1:]
        )
        sample_rv = _SampleRV(rv, factor.function, len(sample_rvs_map), parent_val_indexes, None)
        sample_rvs_map[rv_idx] = sample_rv
        sample_rvs_list.append(sample_rv)
    return sample_rv
