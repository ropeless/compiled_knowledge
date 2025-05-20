from dataclasses import dataclass
from itertools import count
from typing import Callable, Sequence, Optional, Set, Tuple, Dict, Collection

from ck.pgm import Instance, RandomVariable, Indicator
from ck.pgm_circuit.program_with_slotmap import ProgramWithSlotmap
from ck.pgm_circuit.slot_map import SlotMap
from ck.probability.probability_space import Condition, check_condition
from ck.utils.map_set import MapSet
from ck.utils.np_extras import NDArrayStates, NDArrayNumeric
from ck.utils.random_extras import Random

# Type of a yield function. Support for a sampler.
# A yield function may be used to implement a sampler's iterator, thus
# it provides an Instance or single state index.
YieldF = Callable[[NDArrayStates], int] | Callable[[NDArrayStates], Instance]


@dataclass
class SampleRV:
    """
    Support for a sampler.
    A SampleRV structure keeps track of information for one sampled random variable.
    """
    index: int  # index into the sequence of sample rvs.
    rv: RandomVariable  # the random variable being sampled.
    slots: Sequence[int]  # program input slots for indicators of the random variable (co-indexed with rv.states).
    copy_index: Optional[int]  # for Markov chains, which previous sample rv should be copied?


@dataclass
class SamplerInfo:
    """
    Support for a sampler.
    A SamplerInfo structure keeps track of standard information when a sampler uses a Program.
    """
    sample_rvs: Sequence[SampleRV]
    condition: Sequence[Indicator]
    yield_f: YieldF
    slots_0: Set[int]
    slots_1: Set[int]

    @property
    def rvs(self) -> Tuple[RandomVariable, ...]:
        """
        Extract the RandomVariable objects from `self.sample_rvs`.
        """
        return tuple(sample_rv.rv for sample_rv in self.sample_rvs)


def get_sampler_info(
        program_with_slotmap: ProgramWithSlotmap,
        rvs: Optional[RandomVariable | Sequence[RandomVariable]],
        condition: Condition,
        chain_pairs: Sequence[Tuple[RandomVariable, RandomVariable]] = (),
        initial_chain_condition: Condition = (),
) -> SamplerInfo:
    """
    Helper for samplers.

    Determines:
    (1) the slots for sampling rvs,
    (2) Markov chaining rvs,
    (3) the function to use for yielding an Instance or state index.

    If parameter `rvs` is a RandomVariable, then the yield function will
    provide a state index. If parameter `rvs` is a Sequence, then the
    yield function will provide an Instance.

    Args:
        program_with_slotmap: the program and slotmap being referenced.
        rvs: the random variables to sample. It may be either a sequence of
            random variables, or a single random variable.
        condition: is a collection of zero or more conditioning indicators.
        chain_pairs: is a collection of pairs of random variables, each random variable
            must be in the given rvs. Given a pair (from_rv, to_rv) the state of from_rv is used
            as a condition for to_rv prior to generating a sample.
        initial_chain_condition: are condition indicators (just like condition)
            for the initialisation of the 'to_rv' random variables mentioned in chain_pairs.

    Raises:
        ValueError: if preconditions of `program_with_slotmap` are incompatible with the given condition.

    Returns:
        a SamplerInfo structure.
    """
    if rvs is None:
        rvs = program_with_slotmap.rvs
    if isinstance(rvs, RandomVariable):
        # a single rv
        rvs = (rvs,)
        yield_f = lambda x: x.item()
    else:
        # a sequence of rvs
        rvs = tuple(rvs)
        yield_f = lambda x: x.tolist()

    # Group condition indicators by `rv_idx`.
    conditioned_rvs: MapSet[int, Indicator] = MapSet()
    for ind in check_condition(condition):
        conditioned_rvs.add(ind.rv_idx, ind)
    del condition

    # Group precondition indicators by `rv_idx`.
    preconditioned_rvs: MapSet[int, Indicator] = MapSet()
    for ind in program_with_slotmap.precondition:
        preconditioned_rvs.add(ind.rv_idx, ind)

    # Rationalise conditioned_rvs with preconditioned_rvs
    rv_idx: int
    precondition_set: Set[Indicator]
    for rv_idx, precondition_set in preconditioned_rvs.items():
        condition_set = conditioned_rvs.get(rv_idx)
        if condition_set is None:
            # A preconditioned rv was not mentioned in the explicit conditions
            conditioned_rvs.add_all(rv_idx, precondition_set)
        else:
            # A preconditioned rv was also mentioned in the explicit conditions
            condition_set.intersection_update(precondition_set)
            if len(condition_set) == 0:
                rv_index: Dict[int, RandomVariable] = {rv.idx: rv for rv in rvs}
                rv: RandomVariable = rv_index[rv_idx]
                raise ValueError(f'conditions on rv {rv} are disjoint from preconditions')
    del preconditioned_rvs

    # Group initial chain indicators by `rv_idx`.
    initial_chain_condition: Sequence[Indicator] = check_condition(initial_chain_condition)
    initial_chain_conditioned_rvs: MapSet[int, Indicator] = MapSet()
    for ind in initial_chain_condition:
        initial_chain_conditioned_rvs.add(ind.rv_idx, ind)

    # Check sample rvs are valid and without duplicates.
    rvs_set: Set[RandomVariable] = set(rvs)
    if not rvs_set.issubset(program_with_slotmap.rvs):
        raise ValueError('sample random variables not available')
    if len(rvs) != len(rvs_set):
        raise ValueError('duplicate sample random variables requested')

    # Check chain_pairs rvs are being sampled
    if not rvs_set.issuperset(pair[0] for pair in chain_pairs):
        raise ValueError('a random variable appears in chain_pairs but not in sample rvs')
    if not rvs_set.issuperset(pair[1] for pair in chain_pairs):
        raise ValueError('a random variable appears in chain_pairs but not in sample rvs')

    # Check chain_pairs source and destination rvs are disjoint
    if not {pair[0] for pair in chain_pairs}.isdisjoint(pair[1] for pair in chain_pairs):
        raise ValueError('chain_pairs sources and destinations are not disjoint')

    # Check no chain_pairs destination rv is a conditioned rv
    if any(pair[1].idx in conditioned_rvs.keys() for pair in chain_pairs):
        raise ValueError('a chain_pairs destination is conditioned')

    # Check chain initial conditions relate to chain_pairs destination rvs
    chain_dest_rv_idxs: Set[int] = {pair[1].idx for pair in chain_pairs}
    if not all(rv_idx in chain_dest_rv_idxs for rv_idx in initial_chain_conditioned_rvs.keys()):
        raise ValueError('a chain initial condition is not a chain destination rv')

    # Convert chain_pairs for registering with `sample_rvs`.
    # rv_idx maps RandomVariable id to a position it exists in rvs (doesn't matter if rv is duplicated in rvs)
    # copy_idx RandomVariable id to a position in rvs that it can be copied from for Markov chaining.
    rv_idx: Dict[int, int] = {id(rv): i for i, rv in enumerate(rvs)}
    copy_idx: Dict[int, int] = {id(rv): rv_idx[id(prev_rv)] for prev_rv, rv in chain_pairs}

    # Get rv state slots, rvs_slots is co-indexed with rvs
    slot_map: SlotMap = program_with_slotmap.slot_map
    rvs_slots = tuple(tuple(slot_map[ind] for ind in rv) for rv in rvs)

    sample_rvs: Sequence[SampleRV] = tuple(
        SampleRV(idx, rv, rv_slots, copy_idx.get(id(rv)))
        for idx, rv, rv_slots in zip(count(), rvs, rvs_slots)
    )

    # Process the condition to get zero and one slots
    slots_0: Set[int] = set()
    slots_1: Set[int] = set()
    for rv in program_with_slotmap.rvs:
        conditioning: Optional[Set[Indicator]] = conditioned_rvs.get(rv.idx)
        if conditioning is not None:
            slots_1.update(slot_map[ind] for ind in conditioning)
            slots_0.update(slot_map[ind] for ind in rv if ind not in conditioning)
            continue

        conditioning: Optional[Set[Indicator]] = initial_chain_conditioned_rvs.get(rv.idx)
        if conditioning is not None:
            slots_1.update(slot_map[ind] for ind in conditioning)
            slots_0.update(slot_map[ind] for ind in rv if ind not in conditioning)
            continue

        # default
        slots_1.update(slot_map[ind] for ind in rv)

    return SamplerInfo(
        sample_rvs=sample_rvs,
        condition=tuple(ind for condition_set in conditioned_rvs.values() for ind in condition_set),
        yield_f=yield_f,
        slots_0=slots_0,
        slots_1=slots_1,
    )


def uniform_random_sample(
        sample_rvs: Sequence[SampleRV],
        slots_0: Collection[int],
        slots_1: Collection[int],
        slots: NDArrayNumeric,
        state: NDArrayStates,
        rand: Random,
):
    """
    Helper for samplers.

    Sets the states to a random instance and configures slots to match.
    States are drawn from a uniform distribution, drawn using random.randrange.
    """

    # Set up the input slots to respect conditioning
    for slot in slots_0:
        slots[slot] = 0
    for slot in slots_1:
        slots[slot] = 1

    for sample_rv in sample_rvs:
        candidates = []
        for slot_state, slot in enumerate(sample_rv.slots):
            if slots[slot] == 1:
                slots[slot] = 0
                candidates.append((slot_state, slot))

        # Pick a random state for sample_rv
        slot_state, slot = candidates[rand.randrange(0, len(candidates))]
        state[sample_rv.index] = slot_state
        slots[slot] = 1
