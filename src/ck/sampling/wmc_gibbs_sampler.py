from typing import Collection, Iterator, Sequence, List

import numpy as np

from ck.pgm import Instance
from ck.probability.probability_space import dtype_for_state_indexes
from ck.program.program_buffer import ProgramBuffer
from ck.program.raw_program import RawProgram
from ck.sampling.sampler import Sampler
from ck.sampling.sampler_support import SampleRV, YieldF, uniform_random_sample, SamplerInfo
from ck.utils.np_extras import NDArrayStates, NDArrayFloat64
from ck.utils.random_extras import Random, random_permute


class WMCGibbsSampler(Sampler):

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
        self._sample_rvs: List[SampleRV] = list(sampler_info.sample_rvs)
        self._state_dtype = dtype_for_state_indexes(self.rvs)
        self._slots_0: Collection[int] = sampler_info.slots_0
        self._slots_1: Collection[int] = sampler_info.slots_1
        self._skip: int = skip
        self._burn_in: int = burn_in
        self._pr_restart: float = pr_restart

    def __iter__(self) -> Iterator[Instance] | Iterator[int]:
        sample_rvs: List[SampleRV] = self._sample_rvs
        rand: Random = self._rand
        yield_f: YieldF = self._yield_f
        slots_0: Collection[int] = self._slots_0
        slots_1: Collection[int] = self._slots_1
        program_buffer: ProgramBuffer = self._program_buffer
        skip: int = self._skip
        burn_in: int = self._burn_in
        pr_restart: float = self._pr_restart

        # Allocate working memory
        state = np.zeros(len(sample_rvs), dtype=self._state_dtype)
        prs: Sequence[NDArrayFloat64] = tuple(
            np.zeros(len(sample_rv.slots), dtype=np.float64)
            for sample_rv in sample_rvs
        )

        # Set an initial system state
        uniform_random_sample(sample_rvs, slots_0, slots_1, program_buffer.vars, state, rand)

        # Run a burn in
        for i in range(burn_in):
            self._next_sample_gibbs(sample_rvs, slots_1, program_buffer, prs, state, rand)

        if pr_restart <= 0:
            # There is no possibility of a restart
            if skip == 0:
                while True:
                    self._next_sample_gibbs(sample_rvs, slots_1, program_buffer, prs, state, rand)
                    # We know the yield function will always provide either ints or Instances
                    # noinspection PyTypeChecker
                    yield yield_f(state)
            else:
                while True:
                    for _ in range(skip):
                        self._next_sample_gibbs(sample_rvs, slots_1, program_buffer, prs, state, rand)
                    self._next_sample_gibbs(sample_rvs, slots_1, program_buffer, prs, state, rand)
                    # We know the yield function will always provide either ints or Instances
                    # noinspection PyTypeChecker
                    yield yield_f(state)

        else:
            # There is the possibility of a restart
            while True:
                for _ in range(skip):
                    self._next_sample_gibbs(sample_rvs, slots_1, program_buffer, prs, state, rand)
                self._next_sample_gibbs(sample_rvs, slots_1, program_buffer, prs, state, rand)
                # We know the yield function will always provide either ints or Instances
                # noinspection PyTypeChecker
                yield yield_f(state)
                if rand.random() < pr_restart:
                    # Set an initial system state
                    uniform_random_sample(sample_rvs, slots_0, slots_1, program_buffer.vars, state, rand)

                    # Run a burn in
                    for i in range(burn_in):
                        self._next_sample_gibbs(sample_rvs, slots_1, program_buffer, prs, state, rand)

    @staticmethod
    def _next_sample_gibbs(
            sample_rvs: List[SampleRV],
            slots_1: Collection[int],
            program_buffer: ProgramBuffer,
            prs: Sequence[NDArrayFloat64],
            state: NDArrayStates,
            rand: Random
    ) -> None:
        """
        Updates the states to a random system and reconfigures program inputs to match.
        """
        prog_in = program_buffer.vars
        random_permute(sample_rvs, rand=rand)
        for sample_rv in sample_rvs:
            rv_slots = sample_rv.slots
            index = sample_rv.index

            rv_pr: NDArrayFloat64 = prs[index]
            s: int = state.item(index)

            candidates = []
            for slot_state, slot in enumerate(rv_slots):
                if slot in slots_1:
                    candidates.append((slot_state, slot))
            assert len(candidates) > 0

            # Compute conditioned marginals for the current rv
            prog_in[rv_slots[s]] = 0
            for slot_state, slot in candidates:
                prog_in[slot] = 1
                rv_pr[slot_state] = program_buffer.compute()
                prog_in[slot] = 0

            # Pick a new state based on the conditional probabilities
            total = np.sum(rv_pr)
            if total == 0.0:
                # No state of the current rv has a non-zero probability when
                # conditioned on the other random variables states.
                # Pick a random state form a uniform distribution.
                i = rand.randrange(0, len(candidates))
                candidate = candidates[i]
                # update the states array and the wmc input
                state[index] = candidate[0]
                prog_in[candidate[1]] = 1
            else:
                # Pick a state, sampled from the marginal distribution
                r = rand.random() * total
                slot = None
                slot_state = None
                for slot_state, slot in candidates:
                    if r <= rv_pr[slot_state]:
                        break
                    r -= rv_pr[slot_state]
                # update the states array and the wmc input
                state[index] = slot_state
                prog_in[slot] = 1
