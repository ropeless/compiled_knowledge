from typing import Collection, Iterator, Sequence

import numpy as np

from ck.pgm import Instance
from ck.probability.probability_space import dtype_for_state_indexes
from ck.program.program_buffer import ProgramBuffer
from ck.program.raw_program import RawProgram
from ck.sampling.sampler import Sampler
from ck.sampling.sampler_support import SampleRV, YieldF, uniform_random_sample, SamplerInfo
from ck.utils.np_extras import NDArrayStates, DTypeStates
from ck.utils.random_extras import Random


class WMCMetropolisSampler(Sampler):

    def __init__(
            self,
            sampler_info: SamplerInfo,
            raw_program: RawProgram,
            rand: Random,
            skip: int,
            burn_in: int,
            pr_restart: float,
    ):
        super().__init__(sampler_info.rvs, sampler_info.condition)
        self._yield_f: YieldF = sampler_info.yield_f
        self._rand: Random = rand
        self._program_buffer = ProgramBuffer(raw_program)
        self._sample_rvs: Sequence[SampleRV] = tuple(sampler_info.sample_rvs)
        self._state_dtype: DTypeStates = dtype_for_state_indexes(self.rvs)
        self._slots_0: Collection[int] = sampler_info.slots_0
        self._slots_1: Collection[int] = sampler_info.slots_1
        self._skip: int = skip
        self._burn_in: int = burn_in
        self._pr_restart: float = pr_restart

    def __iter__(self) -> Iterator[Instance] | Iterator[int]:
        sample_rvs = self._sample_rvs
        rand = self._rand
        yield_f = self._yield_f
        slots_0 = self._slots_0
        slots_1 = self._slots_1
        program_buffer = self._program_buffer
        slots = program_buffer.vars
        skip = self._skip
        burn_in = self._burn_in
        pr_restart = self._pr_restart

        # Allocate working memory
        state: NDArrayStates = np.zeros(len(sample_rvs), dtype=self._state_dtype)

        # set up the input slots to respect conditioning
        for slot in slots_0:
            slots[slot] = 0
        for slot in slots_1:
            slots[slot] = 1

        # Convert sample slots to possibles
        # And map slots to states.
        possibles = []
        for sample_rv in sample_rvs:
            rv_possibles = []
            for slot_state, slot in enumerate(sample_rv.slots):
                if slots[slot] == 1:
                    rv_possibles.append((slot_state, slot))
            possibles.append((sample_rv.index, sample_rv.slots, rv_possibles))

        # Set an initial valid system state
        w: float = self._init_sample_metropolis(state)

        # Run a burn in
        for i in range(burn_in):
            w = self._next_sample_metropolis(possibles, program_buffer, state, w, rand)

        if pr_restart <= 0:
            # There is no possibility of a restart
            if skip == 0:
                while True:
                    w = self._next_sample_metropolis(possibles, program_buffer, state, w, rand)
                    # We know the yield function will always provide either ints or Instances
                    # noinspection PyTypeChecker
                    yield yield_f(state)
            else:
                while True:
                    for _ in range(skip):
                        w = self._next_sample_metropolis(possibles, program_buffer, state, w, rand)
                    w = self._next_sample_metropolis(possibles, program_buffer, state, w, rand)
                    # We know the yield function will always provide either ints or Instances
                    # noinspection PyTypeChecker
                    yield yield_f(state)

        else:
            # There is the possibility of a restart
            while True:
                for _ in range(skip):
                    w = self._next_sample_metropolis(possibles, program_buffer, state, w, rand)
                w = self._next_sample_metropolis(possibles, program_buffer, state, w, rand)
                # We know the yield function will always provide either ints or Instances
                # noinspection PyTypeChecker
                yield yield_f(state)

                if rand.random() < pr_restart:
                    # Set an initial valid system state
                    w = self._init_sample_metropolis(state)
                    # Run a burn in
                    for i in range(burn_in):
                        w = self._next_sample_metropolis(possibles, program_buffer, state, w, rand)

    def _init_sample_metropolis(self, state: NDArrayStates) -> float:
        """
        Initialises the states to a valid random system and configures program inputs to match.
        """
        sample_rvs = self._sample_rvs
        rand = self._rand
        slots_0 = self._slots_0
        slots_1 = self._slots_1
        program_buffer = self._program_buffer
        slots = program_buffer.vars

        while True:
            uniform_random_sample(sample_rvs, slots_0, slots_1, slots, state, rand)
            w: float = program_buffer.compute().item()
            if w >= 0:
                return w

    @staticmethod
    def _next_sample_metropolis(
            possibles,
            program_buffer: ProgramBuffer,
            state,
            cur_w: float,
            rand: Random,
    ) -> float:
        """
        Updates the states to a random system and reconfigures program inputs to match.
        """
        prog_in = program_buffer.vars

        # Generate a proposal.
        # randomly choose a random variable
        i = rand.randrange(0, len(possibles))
        idx, rv_slots, rv_possibles = possibles[i]
        # keep track of the current state slot
        cur_s = state[idx]
        cur_s_slot = rv_slots[cur_s]
        # randomly choose a possible state
        i = rand.randrange(0, len(rv_possibles))
        s, s_slot = rv_possibles[i]

        # set up state and program to compute weight
        prog_in[cur_s_slot] = 0
        prog_in[s_slot] = 1

        # calculate the weight and test it
        new_w: float = program_buffer.compute().item()
        if rand.random() * cur_w < new_w:
            # accept
            state[idx] = s
            return new_w
        else:
            # reject: set state and program to what it was before
            prog_in[s_slot] = 0
            prog_in[cur_s_slot] = 1
            return cur_w
