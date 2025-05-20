from typing import Collection, Iterator, Sequence

import numpy as np

from ck.pgm import Instance
from ck.probability.probability_space import dtype_for_state_indexes
from ck.program.program_buffer import ProgramBuffer
from ck.program.raw_program import RawProgram
from ck.sampling.sampler import Sampler
from ck.sampling.sampler_support import SampleRV, YieldF, uniform_random_sample, SamplerInfo
from ck.utils.np_extras import NDArrayNumeric
from ck.utils.random_extras import Random


class WMCRejectionSampler(Sampler):

    def __init__(
            self,
            sampler_info: SamplerInfo,
            raw_program: RawProgram,
            rand: Random,
            z: float,
    ):
        super().__init__(sampler_info.rvs, sampler_info.condition)
        self._yield_f: YieldF = sampler_info.yield_f
        self._rand: Random = rand
        self._program_buffer = ProgramBuffer(raw_program)
        self._sample_rvs: Sequence[SampleRV] = tuple(sampler_info.sample_rvs)
        self._state_dtype = dtype_for_state_indexes(self.rvs)
        self._slots_0: Collection[int] = sampler_info.slots_0
        self._slots_1: Collection[int] = sampler_info.slots_1

        # Initialise fields for tracking max_w
        self._w_max = None  # estimated maximum weight for any one world
        self._w_not_seen = z  # z - w_seen
        self._w_high = 0.0  # highest instance wight seen so far
        self._samples = set()  # what samples have we seen

    def __iter__(self) -> Iterator[Instance] | Iterator[int]:
        sample_rvs = self._sample_rvs
        rand = self._rand
        yield_f = self._yield_f
        slots_0 = self._slots_0
        slots_1 = self._slots_1
        program_buffer = self._program_buffer
        slots: NDArrayNumeric = program_buffer.vars

        # Calling wmc() will give the weighted model count for the state of the current input slots.
        def wmc() -> float:
            return program_buffer.compute().item()

        # Allocate working memory to store a possible world
        state: NDArrayNumeric = np.zeros(len(sample_rvs), dtype=self._state_dtype)

        # Initialise w_max to w_max_marginal, if not done yet.
        if self._w_max is None:
            w_max_marginal = self._w_not_seen  # initially set to z, so a 'large' weight

            # Set up the input slots to 0 or 1 to respect conditioning and initial Markov chain states.
            for slot in slots_0:
                slots[slot] = 0
            for slot in slots_1:
                slots[slot] = 1

            # Loop over the rvs
            for sample_rv in sample_rvs:
                rv_slots = sample_rv.slots
                max_for_rv = 0
                # Set all rv slots to 0
                for slot_state, slot in enumerate(rv_slots):
                    slots[slot] = 0
                back_to_one = []
                # Loop over state of the rv.
                for slot_state, slot in enumerate(rv_slots):
                    if slot in slots_1:
                        slots[slot] = 1
                        w: float = wmc()
                        max_for_rv = max(max_for_rv, w)
                        slots[slot] = 0
                        back_to_one.append(slot)
                # Set rv slots back to 1 as needed (ready for next rv).
                for slot in back_to_one:
                    slots[slot] = 1

                w_max_marginal = min(w_max_marginal, max_for_rv)

            self._w_max = w_max_marginal

        while True:
            uniform_random_sample(sample_rvs, slots_0, slots_1, slots, state, rand)
            w: float = wmc()

            if rand.random() * self._w_max < w:
                # We know the yield function will always provide either ints or Instances
                # noinspection PyTypeChecker
                yield yield_f(state)

            # Update w_not_seen and w_high to adapt w_max.
            # We don't bother tracking seen samples once w_not_seen and w_high
            # are close enough, or we have tracked too many samples.
            if self._samples is not None:
                s = tuple(state)
                if s not in self._samples:
                    self._samples.add(s)
                    self._w_not_seen -= w
                    self._w_high = max(self._w_high, w)
                    w_max_tracked = max(self._w_high, self._w_not_seen)
                    self._w_max = min(w_max_tracked, self._w_max)

                    # See if we should stop tracking samples.
                    if (
                            self._w_not_seen - self._w_high < 0.001  # w_not_seen and w_high are close enough
                            or len(self._samples) > 1000000  # tracked too many samples
                    ):
                        self._samples = None
