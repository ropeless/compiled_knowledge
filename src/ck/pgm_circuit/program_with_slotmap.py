from typing import Tuple, Sequence, Dict, Iterable

from ck.pgm import RandomVariable, rv_instances, Instance, rv_instances_as_indicators, Indicator, ParamId
from ck.pgm_circuit.slot_map import SlotMap, SlotKey
from ck.probability.probability_space import Condition, check_condition
from ck.program.program_buffer import ProgramBuffer
from ck.utils.np_extras import NDArray, NDArrayNumeric


class ProgramWithSlotmap:
    """
    A class for bundling a program buffer with a slot-map, where the slot-map maps keys
    (e.g., random variable indicators) to program input slots.
    """

    def __init__(
            self,
            program_buffer: ProgramBuffer,
            slot_map: SlotMap,
            rvs: Sequence[RandomVariable],
            precondition: Sequence[Indicator]
    ):
        """
        Construct a ProgramWithSlotmap object.

        Args:
            program_buffer: is a ProgramBuffer object which is a compiled circuit with input and output slots.
            slot_map: a maps from a slot_key to input slot of 'program'.
            rvs: a sequence of rvs used for setting program input slots, each rv
                has a length and rv[i] is a unique 'indicator' across all rvs.
            precondition: conditions on rvs that are compiled into the program.

        Raises:
            ValueError: if rvs contains duplicates.
        """
        self._program_buffer: ProgramBuffer = program_buffer
        self._slot_map: SlotMap = slot_map
        self._rvs: Tuple[RandomVariable, ...] = tuple(rvs)
        self._precondition: Tuple[Indicator, ...] = tuple(precondition)

        if len(rvs) != len(set(rv.idx for rv in rvs)):
            raise ValueError('duplicate random variables provided')

        # Given rv = rvs[i], then _rvs_slots[i][state_idx] gives the slot for rv[state_idx].
        self._rvs_slots: Tuple[Tuple[int, ...], ...] = tuple(tuple(self._slot_map[ind] for ind in rv) for rv in rvs)

        # Given rv = rvs[i], then _indicator_map maps[rv[j]] = (i, slot), where slot is for indicator rv[j].
        self._indicator_map: Dict[Indicator, Tuple[int, int]] = {
            ind: (i, slot_map[ind])
            for i, rv in enumerate(rvs)
            for ind in rv
        }

    @property
    def rvs(self) -> Sequence[RandomVariable]:
        """
        What are the random variables considered as 'inputs'.
        """
        return self._rvs

    @property
    def precondition(self) -> Sequence[Indicator]:
        """
        Condition on `self.rvs` that is compiled into the program.
        """
        return self._precondition

    @property
    def slot_map(self) -> SlotMap:
        return self._slot_map

    def instances(self, flip: bool = False) -> Iterable[Instance]:
        """
        Enumerate instances of the random variables.

        Each instance is a tuples of state indexes, co-indexed with the given random variables.

        The order is the natural index order (i.e., last random variable changing most quickly).

        Args:
            flip: if true, then first random variable changes most quickly.

        Returns:
            an iteration over tuples, each tuple holds state indexes
            co-indexed with the given random variables.
        """
        return rv_instances(*self._rvs, flip=flip)

    def instances_as_indicators(self, flip: bool = False) -> Iterable[Sequence[Indicator]]:
        """
        Enumerate instances of the random variables.

        Each instance is a tuples of indicators, co-indexed with the given random variables.

        The order is the natural index order (i.e., last random variable changing most quickly).

        Args:
            flip: if true, then first random variable changes most quickly.

        Returns:
            an iteration over tuples, each tuples holds random variable indicators
            co-indexed with the given random variables.
        """
        return rv_instances_as_indicators(*self._rvs, flip=flip)

    def compute(self) -> NDArrayNumeric:
        """
        Execute the program to compute and return the result. As per `ProgramBuffer.compute`.

        Warning:
            when returning an array, the array is backed by the program buffer memory, not a copy.
        """
        return self._program_buffer.compute()

    def compute_conditioned(self, *condition: Condition) -> NDArrayNumeric:
        """
        Equivalent to:
            self.set_condition(*condition)
            return self.compute()
        """
        self.set_condition(*condition)
        return self.compute()

    @property
    def results(self) -> NDArrayNumeric:
        """
        Get the results of the last computation.
        As per `ProgramBuffer.results`.

        Warning:
            the array is backed by the program buffer memory, not a copy.
        """
        return self._program_buffer.results

    @property
    def vars(self) -> NDArrayNumeric:
        """
        Return the input variables as a numpy array.
        As per `ProgramBuffer.vars`.

        Warning:
            writing to the returned array will write to the input slots of the program buffer.
        """
        return self._program_buffer.vars

    def __setitem__(self, item: int | slice | SlotKey | Iterable[SlotKey], value: float) -> None:
        """
        Set one or more input slot values, identified by slot keys.
        """
        if isinstance(item, (int, slice)):
            self._program_buffer[item] = value
        elif isinstance(item, (Indicator, ParamId)):
            self._program_buffer[self._slot_map[item]] = value
        else:
            # Assume its iterable
            for i in item:
                self[i] = value

    def __getitem__(self, item: int | slice | SlotKey) -> NDArrayNumeric:
        """
        Get an input slot value, identified by a slot key.
        """
        if isinstance(item, (int, slice)):
            return self._program_buffer[item]
        elif isinstance(item, (Indicator, ParamId)):
            return self._program_buffer[self._slot_map[item]]
        else:
            raise IndexError('unknown index type')

    def set_condition(self, *condition: Condition) -> None:
        """
        Set the input slots of random variables to 1, except where implied to
        0 according to the given conditions.

        Specifically:
            each slot corresponding to an indicator given condition will be set to 1;

            if a random variable is mentioned in the given indicators, then all
            slots for indicators for that random variable, except for slots corresponding
            to an indicator given condition;

            if a random variable is not mentioned in the given condition, that random variable
            will have all its slots set to 1.
        """
        condition: Sequence[Indicator] = check_condition(condition)

        ind_slot_groups = [[] for _ in self._rvs_slots]
        for ind in condition:
            rv_idx, slot = self._indicator_map[ind]
            ind_slot_groups[rv_idx].append(slot)

        slots: NDArray = self._program_buffer.vars
        for rv_slots, ind_slots in zip(self._rvs_slots, ind_slot_groups):
            if len(ind_slots) == 0:
                # this rv _is not_ mentioned in the indicators - marginalise it
                for slot in rv_slots:
                    slots[slot] = 1
            else:
                # this rv _is_ mentioned in the indicators - we set the mentioned slots to 1 and others to 0.
                for slot in rv_slots:
                    slots[slot] = 0
                for slot in ind_slots:
                    slots[slot] = 1

    def set_rv(self, rv: RandomVariable, *values: float | int) -> None:
        """
        Set the input values of a random variable.

        Args:
            rv: a random variable whose indicators are in the slot map.
            values: list of values, assumes len(values) == len(rv).
        """
        for i in range(len(rv)):
            self[rv[i]] = values[i]

    def set_rvs_uniform(self, *rvs: RandomVariable) -> None:
        """
        Set the input values for each rv in rvs to 1 / len(rv).

        Args:
            rvs: a collection of random variable whose indicators are in the slot map.
        """
        for rv in rvs:
            value = 1.0 / len(rv)
            for ind in rv:
                self[ind] = value

    def set_all_rvs_uniform(self) -> None:
        """
        Set the input values for each rv in rvs to 1 / len(rv).
        """
        slots: NDArray = self._program_buffer.vars
        for rv_slots in self._rvs_slots:
            value = 1.0 / len(rv_slots)
            for slot in rv_slots:
                slots[slot] = value
