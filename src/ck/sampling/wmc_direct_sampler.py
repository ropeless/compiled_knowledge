from typing import Collection, Iterator, Sequence

import numpy as np

from ck.pgm import Instance
from ck.probability.probability_space import dtype_for_state_indexes
from ck.program.program_buffer import ProgramBuffer
from ck.program.raw_program import RawProgram
from ck.sampling.sampler import Sampler
from ck.sampling.sampler_support import SampleRV, YieldF, SamplerInfo
from ck.utils.np_extras import NDArrayNumeric, NDArrayStates
from ck.utils.random_extras import Random


class WMCDirectSampler(Sampler):

    def __init__(
            self,
            sampler_info: SamplerInfo,
            raw_program: RawProgram,
            rand: Random,
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

        # Set up the input slots to 0 or 1 to respect conditioning and initial Markov chain states.
        slots: NDArrayNumeric = self._program_buffer.vars
        for slot in sampler_info.slots_0:
            slots[slot] = 0
        for slot in sampler_info.slots_1:
            slots[slot] = 1

    def __iter__(self) -> Iterator[Instance] | Iterator[int]:
        yield_f = self._yield_f
        rand = self._rand
        sample_rvs = self._sample_rvs
        chain_rvs = self._chain_rvs
        slots_1 = self._slots_1
        program_buffer = self._program_buffer
        slots: NDArrayNumeric = program_buffer.vars

        # Calling wmc() will give the weighted model count for the state of the current input slots.
        def wmc() -> float:
            return program_buffer.compute().item()

        # Set up working memory buffers
        states: NDArrayStates = np.zeros(len(sample_rvs), dtype=self._state_dtype)
        buff_slots = np.zeros(self._max_number_of_states, dtype=np.uintp)
        buff_states = np.zeros(self._max_number_of_states, dtype=self._state_dtype)

        while True:
            # Consider all possible instantiations given the conditions, c, where the instantiations are ordered.
            # Let awmc(i|c) be the accumulated WMC of the ith instantiation.
            # We want to find the smallest instantiation i such that
            #     rnd <= awmc(i|c)
            # where rnd is in [0, 1) * wmc().

            rnd: float = rand.random() * wmc()

            for sample_rv in sample_rvs:
                # Prepare to loop over random variable states.
                # Keep track of the non-zero slots in buff_slots and buff_states.
                num_possible_states: int = 0
                for j, slot in enumerate(sample_rv.slots):
                    if slots[slot] != 0:
                        buff_slots[num_possible_states] = slot
                        buff_states[num_possible_states] = j
                        num_possible_states += 1

                if num_possible_states == 0:
                    raise RuntimeError('zero probability')

                # Try each possible state of the current random variable.
                # Once a state is selected, then the following is true:
                #   states[rv_position] = state
                #   m_prev_states[rv_position] = state
                #   slots set up to include condition rv = state.
                #   rnd is reduced to account for the states skipped.
                #
                # We can do this either by sequentially checking each state or by doing
                # a binary search. Here we start with binary search then finish sequentially
                # once the candidates size falls below 'THRESHOLD'.

                # Binary search
                THRESHOLD = 2
                lo: int = 0
                hi: int = num_possible_states
                w_0_mark: int = 0
                w: float = 0
                while lo + THRESHOLD < hi:
                    mid: int = (lo + hi) // 2

                    for i in range(mid, hi):
                        slots[buff_slots[i]] = 0

                    w = wmc()
                    w_0_mark = mid
                    if w < rnd:
                        # wmc() is too low, the desired state is >= buff_states[mid]
                        for i in range(mid, hi):
                            slots[buff_slots[i]] = 1
                        lo = mid
                    else:
                        # wmc() is too high, the desired state is < buff_states[mid]
                        hi = mid

                # Now the state we want is between lo (inclusive) and hi (exclusive).
                # Slots at least up to lo will be set to 1.

                # clear top slots, lo and up.
                for k in range(lo, num_possible_states):
                    slots[buff_slots[k]] = 0

                # Adjust rnd to account for lo > 0.
                if lo == 0:
                    # The chances of this case may be low, but if so, then
                    # slots[m_buff_slots[lo]] = 0  which implies wmc() == 0,
                    # so we can save a call to wmc().
                    pass
                elif w_0_mark == lo:
                    # We can use the last wmc() call, stored in w.
                    # This saves a call to wmc().
                    rnd -= w
                else:
                    rnd -= wmc()

                # Clear remaining slots
                for k in range(0, lo):
                    slots[buff_slots[k]] = 0

                # Sequential search
                k = lo
                while k < hi:
                    slot = buff_slots[k]
                    slots[slot] = 1
                    w = wmc()
                    if rnd < w:
                        break
                    slots[slot] = 0
                    rnd -= w
                    k += 1

                slot = buff_slots[k]
                state = buff_states[k]
                slots[slot] = 1
                states[sample_rv.index] = state

            # We know the yield function will always provide either ints or Instances
            # noinspection PyTypeChecker
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
