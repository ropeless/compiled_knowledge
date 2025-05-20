import random
from typing import Set, List, Iterator, Optional, Sequence

import numpy as np

from ck.pgm import Instance, RandomVariable, Indicator
from ck.probability.probability_space import dtype_for_state_indexes, Condition, check_condition
from ck.utils.map_set import MapSet
from ck.utils.np_extras import DType
from ck.utils.random_extras import Random
from .sampler import Sampler
from .sampler_support import YieldF


class UniformSampler(Sampler):

    def __init__(
            self,
            rvs: RandomVariable | Sequence[RandomVariable],
            condition: Condition = (),
            rand: Random = random,
    ):
        condition: Sequence[Indicator] = check_condition(condition)

        self._yield_f: YieldF
        if isinstance(rvs, RandomVariable):
            # a single rv
            rvs = (rvs,)
            self._yield_f = lambda x: x.item()
        else:
            # a sequence of rvs
            self._yield_f = lambda x: x.tolist()

        super().__init__(rvs, condition)

        # Group condition indicators by `rv_idx`.
        conditioned_rvs: MapSet[int, int] = MapSet()
        for ind in condition:
            conditioned_rvs.add(ind.rv_idx, ind.state_idx)

        def get_possible_states(_rv: RandomVariable) -> List[int]:
            """
            Get the allowable states for a given random variable, given
            conditions in `conditioned_rvs`.
            """
            condition_states: Optional[Set[int]] = conditioned_rvs.get(_rv.idx)
            if condition_states is None:
                return list(range(len(_rv)))
            else:
                return list(condition_states)

        possible_states: List[List[int]] = [
            get_possible_states(rv)
            for rv in self.rvs
        ]

        self._possible_states: List[List[int]] = possible_states
        self._rand: Random = rand
        self._state_dtype: DType = dtype_for_state_indexes(self.rvs)

    def __iter__(self) -> Iterator[Instance] | Iterator[int]:
        possible_states = self._possible_states
        yield_f = self._yield_f
        rand = self._rand
        state = np.zeros(len(possible_states), dtype=self._state_dtype)
        while True:
            for i, l in enumerate(possible_states):
                state_idx = rand.randrange(0, len(l))
                state[i] = l[state_idx]
            # We know the yield function will always provide either ints or Instances
            # noinspection PyTypeChecker
            yield yield_f(state)
