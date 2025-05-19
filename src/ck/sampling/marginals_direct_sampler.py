from typing import Collection, Iterator, Dict, Sequence

import numpy as np

from ck.pgm import Instance
from ck.probability.probability_space import dtype_for_state_indexes
from ck.program.program_buffer import ProgramBuffer
from ck.program.raw_program import RawProgram
from ck.sampling.sampler import Sampler
from ck.sampling.sampler_support import SampleRV, YieldF, SamplerInfo
from ck.utils.np_extras import NDArray, NDArrayNumeric
from ck.utils.random_extras import Random


class MarginalsDirectSampler(Sampler):

    def __init__(
            self,
            sampler_info: SamplerInfo,
            raw_program: RawProgram,
            rand: Random,
            rv_idx_to_result_offset: Dict[int, int],
    ):
        super().__init__(sampler_info.rvs, sampler_info.condition)
        self._yield_f: YieldF = sampler_info.yield_f
        self._rand: Random = rand
        self._program_buffer = ProgramBuffer(raw_program)
        self._sample_rvs: Sequence[SampleRV] = tuple(sampler_info.sample_rvs)
        self._chain_rvs: Sequence[SampleRV] = tuple(
            sample_rv for sample_rv in sampler_info.sample_rvs if sample_rv.copy_index is not None)
        self._state_dtype = dtype_for_state_indexes(self.rvs)
        self._max_number_of_states: int = max((len(rv) for rv in self.rvs), default=0)
        self._slots_1: Collection[int] = sampler_info.slots_1

        self._marginals: Sequence[NDArrayNumeric] = tuple(
            self._program_buffer.results[
            rv_idx_to_result_offset[sample_rv.rv.idx]
            :
            rv_idx_to_result_offset[sample_rv.rv.idx] + len(sample_rv.rv)
            ]
            for sample_rv in sampler_info.sample_rvs
        )
        # Set up the input slots to 0 or 1 to respect conditioning and initial Markov chain states.
        slots: NDArray = self._program_buffer.vars
        for slot in sampler_info.slots_0:
            slots[slot] = 0
        for slot in sampler_info.slots_1:
            slots[slot] = 1

    def __iter__(self) -> Iterator[Instance] | Iterator[int]:
        yield_f = self._yield_f
        rand = self._rand
        sample_rvs = self._sample_rvs
        chain_rvs = self._chain_rvs
        program_buffer = self._program_buffer
        slots: NDArray = program_buffer.vars
        marginals = self._marginals
        slots_1 = self._slots_1

        # Set up working memory buffer
        states = np.zeros(len(sample_rvs), dtype=self._state_dtype)

        def compute() -> float:
            # Compute the program results based on the current input slot values.
            # Return the WMC.
            return program_buffer.compute().item(-1)

        while True:
            wmc: float = compute()
            rnd: float = rand.random() * wmc

            for sample_rv in sample_rvs:
                index: int = sample_rv.index
                if index > 0:
                    # No need to execute the program on the first time through
                    # as it was done just before entering the loop.
                    wmc = compute()

                rv_dist: NDArray = marginals[sample_rv.index]

                rv_dist_sum: float = rv_dist.sum()
                if rv_dist_sum <= 0:
                    raise RuntimeError('zero probability')
                rv_dist *= wmc / rv_dist_sum

                state_index: int = -1
                for i in range(len(sample_rv.rv)):
                    w = rv_dist.item(i)
                    if rnd < w:
                        state_index = i
                        break
                    rnd -= w
                assert state_index >= 0

                for slot in sample_rv.slots:
                    slots[slot] = 0
                slots[sample_rv.slots[state_index]] = 1
                states[index] = state_index

            yield yield_f(states)

            # Reset the one slots for the next iteration.
            for slot in slots_1:
                slots[slot] = 1

            # Copy chain pairs for next iteration.
            # (This writes over any initial chain conditions from slots_1.)
            for sample_rv in chain_rvs:
                rv_slots = sample_rv.slots
                prev_state_idx: int = states.item(sample_rv.copy_index)
                for slot in rv_slots:
                    slots[slot] = 0
                slots[rv_slots[prev_state_idx]] = 1
