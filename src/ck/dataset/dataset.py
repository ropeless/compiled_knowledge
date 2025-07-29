from __future__ import annotations

from itertools import repeat
from typing import Sequence, Optional, Dict, Iterable, Tuple, List, Iterator

import numpy as np

from ck.pgm import RandomVariable, State, Instance
from ck.utils.np_extras import DTypeStates, dtype_for_number_of_states, NDArrayNumeric, NDArrayStates


class Dataset:
    """
    A dataset has instances (rows) for zero or more random variables.
    Each instance has a weight, which is notionally one.
    Weights of instances should be non-negative, and are normally positive.
    """

    def __init__(
            self,
            weights: Optional[NDArrayNumeric | Sequence],
            length: Optional[int],
    ):
        # Infer the length of the dataset.
        if length is not None:
            self._length: int = length
        else:
            self._length: int = len(weights)

        # Set no random variables
        self._rvs: Tuple[RandomVariable, ...] = ()

        # Set the weights array, and confirm its shape
        self._weights: NDArrayNumeric
        if weights is None:
            weights = np.ones(self._length)
        elif not isinstance(weights, np.ndarray):
            weights = np.array(weights, dtype=np.float64)
        expected_shape = (self._length,)
        if weights.shape != expected_shape:
            raise ValueError(f'weights expected shape {expected_shape}, got {weights.shape}')
        # if not isinstance(weights.dtype, NDArrayNumeric):
        #     raise ValueError('weights expected numeric dtype')

        self._weights = weights

    def __len__(self) -> int:
        """
        How many instances in the dataset.
        """
        return self._length

    @property
    def rvs(self) -> Sequence[RandomVariable]:
        """
        Return the random variables covered by this dataset.
        """
        return self._rvs

    @property
    def weights(self) -> NDArrayNumeric:
        """
        Get the instance weights.
        The notional weight of an instance is 1.
        The index into the returned array is the instance index.

        Returns:
            A 1D array of random variable states, with shape = `(len(self), )`.
        """
        return self._weights

    def total_weight(self) -> float:
        """
        Calculate the total weight of this dataset.
        """
        return self._weights.sum().item()

    def _add_rv(self, rv: RandomVariable) -> None:
        """
        Add a random variable to self.rvs.
        """
        self._rvs += (rv,)

    def _remove_rv(self, rv: RandomVariable) -> None:
        """
        Remove a random variable from self.rvs.
        """
        rvs = self._rvs
        i: int = self._rvs.index(rv)
        self._rvs = rvs[:i] + rvs[i + 1:]


class HardDataset(Dataset):
    """
    A hard dataset is a dataset where for each instance (row) and each random variable,
    there is a state for that random variable (a state is represented as a state index).
    Each instance has a weight, which is notionally one.
    """

    @staticmethod
    def from_soft_dataset(
            soft_dataset: SoftDataset,
            *,
            adjust_instance_weights: bool = True,
    ) -> HardDataset:
        """
        Create a hard dataset from a soft dataset by repeated application
        of `SoftDataset.add_rv_from_state_weights`.

        The instance weights of the returned dataset will be a copy
        of the instance weights of the soft dataset.

        Args:
            soft_dataset: The soft dataset providing random variables,
                their states, and instance weights.
            adjust_instance_weights: If `True` (default), then the instance weights will be
                adjusted according to sum of state weights for each instance. That is, if
                the sum is not one for some instance, then the weight of that instance will
                be adjusted.

        Returns:
            A `HardDataset` instance.
        """
        dataset = HardDataset(weights=soft_dataset.weights.copy())
        for rv in soft_dataset.rvs:
            dataset.add_rv_from_state_weights(rv, soft_dataset.state_weights(rv), adjust_instance_weights)
        return dataset

    def __init__(
            self,
            data: Iterable[Tuple[RandomVariable, NDArrayStates | Sequence[int]]] = (),
            *,
            weights: Optional[NDArrayNumeric | Sequence[float | int]] = None,
            length: Optional[int] = None,
    ):
        """
        Create a hard dataset.

        When `weights` is a numpy array, then the dataset will directly reference the given array.
        When `data` contains a numpy array, then the dataset will directly reference the given array.

        Args:
            data: optional iterable of (random variable, state idxs), passed
                to `self.add_rv_from_state_idxs`.
            weights: optional array of instance weights.
            length: optional length of the dataset, if omitted, the length is inferred.
        """
        self._data: Dict[RandomVariable, NDArrayStates] = {}

        # Initialise super by either weights, length or first data item.
        super_initialised: bool = False
        if weights is not None or length is not None:
            super().__init__(weights, length)
            super_initialised = True

        for rv, states in data:
            if not super_initialised:
                super().__init__(weights, len(states))
                super_initialised = True
            self.add_rv_from_state_idxs(rv, states)

        if not super_initialised:
            super().__init__(weights, 0)

    def state_idxs(self, rv: RandomVariable) -> NDArrayStates:
        """
        Get the state indexes for one random variable.
        The index into the returned array is the instance index.

        Returns:
            A 1D array of random variable states, with shape = `(len(self), )`.

        Raises:
            KeyError: If the random variable is not in the dataset.
        """
        return self._data[rv]

    def add_rv(self, rv: RandomVariable) -> NDArrayStates:
        """
        Add a random variable to the dataset, allocating and returning
        the state indices for the random variable.

        Args:
            rv: The random variable to add.

        Returns:
            A 1D array of random variable states, with shape = `(len(self), )`, initialised to zero.

        Raises:
            ValueError: If the random variable is already in the dataset.
        """
        dtype: DTypeStates = dtype_for_number_of_states(len(rv))
        rv_data = np.zeros(len(self), dtype=dtype)
        return self.add_rv_from_state_idxs(rv, rv_data)

    def remove_rv(self, rv: RandomVariable) -> None:
        """
        Remove a random variable from the dataset.

        Args:
            rv: The random variable to remove.

        Raises:
            KeyError: If the random variable is not in the dataset.
        """
        del self._data[rv]
        self._remove_rv(rv)

    def add_rv_from_state_idxs(self, rv: RandomVariable, state_idxs: NDArrayStates | Sequence[int]) -> NDArrayStates:
        """
        Add a random variable to the dataset.

        When `state_idxs` is a numpy array, then the dataset will directly reference the given array.

        Args:
            rv: The random variable to add.
            state_idxs: An 1D array of state indexes to add, with shape = `(len(self),)`.
                Each element `state` should be `0 <= state < len(rv)`.

        Returns:
            A 1D array of random variable states, with shape = `(len(self), )`.

        Raises:
            ValueError: If the random variable is already in the dataset.
        """
        if rv in self._data.keys():
            raise ValueError(f'data for {rv} already exists in the dataset')

        if isinstance(state_idxs, np.ndarray):
            expected_shape = (self._length,)
            if state_idxs.shape == expected_shape:
                rv_data = state_idxs
            else:
                raise ValueError(f'data for {rv} expected shape {expected_shape}, got {state_idxs.shape}')
        else:
            dtype: DTypeStates = dtype_for_number_of_states(len(rv))
            if len(state_idxs) != self._length:
                raise ValueError(f'data for {rv} expected length {self._length}, got {len(state_idxs)}')
            rv_data = np.array(state_idxs, dtype=dtype)

        self._data[rv] = rv_data
        self._add_rv(rv)
        return rv_data

    def add_rv_from_states(self, rv: RandomVariable, states: Sequence[State]) -> NDArrayStates:
        """
        Add a random variable to the dataset.

        The dataset will allocate and populate a states array containing state indexes.
        This will call `rv.state_idx(state)` for each state in `states`.

        Args:
            rv: The random variable to add.
            states: An 1D array of state to add, with `len(states)` = `len(self)`.
                Each element `state` should be in `rv.states`.

        Returns:
            A 1D array of random variable states, with shape = `(len(self), )`.

        Raises:
            ValueError: If the random variable is already in the dataset.
        """
        dtype: DTypeStates = dtype_for_number_of_states(len(rv))
        rv_data = np.fromiter(
            iter=(
                rv.state_idx(state)
                for state in states
            ),
            dtype=dtype,
            count=len(states)
        )
        return self.add_rv_from_state_idxs(rv, rv_data)

    def add_rv_from_state_weights(
            self,
            rv: RandomVariable,
            state_weights: NDArrayNumeric,
            adjust_instance_weights: bool = True,
    ) -> NDArrayStates:
        """
        Add a random variable to the dataset.

        The dataset will allocate and populate a states array containing state indexes.
        For each instance, the state with the highest weight will be taken to be the
        state of the random variable, with ties broken arbitrarily.

        Args:
            rv: The random variable to add.
            state_weights: An 2D array of state weights, with shape = `(len(self), len(rv))`.
                Each element `state` should be in `rv.states`.
            adjust_instance_weights: If `True` (default), then the instance weights will be
                adjusted according to sum of state weights for each instance. That is, if
                the sum is not one for some instance, then the weight of that instance will
                be adjusted.

        Returns:
            A 1D array of random variable states, with shape = `(len(self), )`.

        Raises:
            ValueError: If the random variable is already in the dataset.
        """
        expected_shape = (self._length, len(rv))
        if state_weights.shape != expected_shape:
            raise ValueError(f'data for {rv} expected shape {expected_shape}, got {state_weights.shape}')

        dtype: DTypeStates = dtype_for_number_of_states(len(rv))
        rv_data = np.fromiter(
            iter=(
                np.argmax(row)
                for row in state_weights
            ),
            dtype=dtype,
            count=self._length
        )

        if adjust_instance_weights:
            row: NDArrayNumeric
            for i, row in enumerate(state_weights):
                self._weights[i] *= row.sum()

        return self.add_rv_from_state_idxs(rv, rv_data)

    def instances(self, rvs: Optional[Sequence[RandomVariable]] = None) -> Iterator[Tuple[Instance, float]]:
        """
        Iterate over weighted instances.

        Args:
            rvs: The random variables to include in iteration. Default is all dataset random variables.

        Returns:
            an iterator over (instance, weight) pairs, in the same order and number of instances in this dataset.
            An instance is a sequence of state indexes, co-indexed with `self.rvs`.
        """
        if rvs is None:
            rvs = self._rvs
        # Special case - no random variables
        if len(rvs) == 0:
            return zip(repeat(()), self.weights)
        else:
            cols = [self.state_idxs(rv) for rv in rvs]
            return zip(zip(*cols), self.weights)

    def dump(self, *, show_rvs: bool = True, show_weights: bool = True, as_states: bool = False) -> None:
        """
        Dump the dataset in a human-readable format.
        If as_states is true, then instance states are dumped instead of just state indexes.

        Args:
            show_rvs: If `True`, the random variables are dumped.
            show_weights: If `True`, the instance weights are dumped.
            as_states: If `True`, the states are dumped instead of just state indexes.
        """
        if show_rvs:
            rvs = ', '.join(str(rv) for rv in self.rvs)
            print(f'rvs: [{rvs}]')
        print(f'instances ({len(self)}, with total weight {self.total_weight()}):')
        for instance, weight in self.instances():
            if as_states:
                instance_str = ', '.join(repr(rv.states[idx]) for idx, rv in zip(instance, self.rvs))
            else:
                instance_str = ', '.join(str(idx) for idx in instance)
            if show_weights:
                print(f'({instance_str}) * {weight}')
            else:
                print(f'({instance_str})')


class SoftDataset(Dataset):
    """
    A soft dataset is a dataset where for each instance (row) and each random variable,
    there is a distribution over the states of that random variable. That is,
    for each instance, for each indicator, there is a weight. Additionally,
    each instance has a weight.

    Weights of random variable states are expected to be non-negative.
    Notionally, the sum of weights for an instance and random variable is one.
    """

    @staticmethod
    def from_hard_dataset(hard_dataset: HardDataset) -> SoftDataset:
        """
        Create a soft dataset from a hard dataset by repeated application
        of `SoftDataset.add_rv_from_state_idxs`.

        The instance weights of the returned dataset will be a copy
        of the instance weights of the hard dataset.

        Args:
            hard_dataset: The hard dataset providing random variables,
                their states, and instance weights.

        Returns:
            A `SoftDataset` instance.
        """
        dataset = SoftDataset(weights=hard_dataset.weights.copy())
        for rv in hard_dataset.rvs:
            dataset.add_rv_from_state_idxs(rv, hard_dataset.state_idxs(rv))
        return dataset

    def __init__(
            self,
            data: Iterable[Tuple[RandomVariable, NDArrayNumeric | Sequence[Sequence[float]]]] = (),
            *,
            weights: Optional[NDArrayNumeric | Sequence[float | int]] = None,
            length: Optional[int] = None,
    ):
        """
        Create a soft dataset.

        When `weights` is a numpy array, then the dataset will directly reference the given array.
        When `data` contains a numpy array, then the dataset will directly reference the given array.

        Args:
            data: optional iterable of (random variable, state weights), passed
                to `self.add_rv_from_state_weights`.
            weights: optional array of instance weights.
            length: optional length of the dataset, if omitted, the length is inferred.
        """
        self._data: Dict[RandomVariable, NDArrayNumeric] = {}

        # Initialise super by either weights, length or first data item.
        super_initialised: bool = False
        if weights is not None or length is not None:
            super().__init__(weights, length)
            super_initialised = True

        for rv, states_weights in data:
            if not super_initialised:
                super().__init__(weights, len(states_weights))
                super_initialised = True
            self.add_rv_from_state_weights(rv, states_weights)

        if not super_initialised:
            super().__init__(weights, 0)

    def normalise(self, check_negative_instance: bool = True) -> None:
        """
        Adjust weights (for states and instances) so that the sum of state weights
        for any random variable is 1 (or zero).

        This performs an in-place modification.

        If an instance weight is zero then all state weights for that instance will be zero.
        If the state weights of an instance for any random variable sum to zero, then
        that instance weight will be zero.

        All other state weights of an instance for each random variable will sum to one.

        Args:
            check_negative_instance: if true (the default),then a RuntimeError is
                raised if a negative instance weight is encountered.

        Raises:
            RuntimeError: if `check_negative_instance` is true and a negative
                instance weight is encountered.
        """
        state_weights: NDArrayNumeric
        i: int

        weights: NDArrayNumeric = self.weights
        for i in range(self._length):
            for state_weights in self._data.values():
                weight_sum = state_weights[i].sum()
                if weight_sum == 0:
                    weights[i] = 0
                elif weight_sum != 1:
                    state_weights[i] /= weight_sum
                    weights[i] *= weight_sum
            instance_weight = weights[i]
            if instance_weight == 0:
                for state_weights in self._data.values():
                    state_weights[i, :] = 0
            elif check_negative_instance and instance_weight < 0:
                raise RuntimeError(f'negative instance weight: {i}')

    def state_weights(self, rv: RandomVariable) -> NDArrayNumeric:
        """
        Get the state weights for one random variable.
        The first index into the returned array is the instance index.
        The second index into the returned array is the state index.

        Returns:
            A 2D array of random variable states, with shape = `(len(self), len(rv))`.

        Raises:
            KeyError: If the random variable is not in the dataset.
        """
        return self._data[rv]

    def add_rv(self, rv: RandomVariable) -> NDArrayNumeric:
        """
        Add a random variable to the dataset, allocating and returning
        the state indices for the random variable.

        Args:
            rv: The random variable to add.

        Returns:
            A 2D array of random variable states, with shape = `(len(self), len(rv))`,
            initialised to zero.

        Raises:
            ValueError: If the random variable is already in the dataset.
        """
        rv_data = np.zeros((len(self), len(rv)), dtype=np.float64)
        return self.add_rv_from_state_weights(rv, rv_data)

    def remove_rv(self, rv: RandomVariable) -> None:
        """
        Remove a random variable from the dataset.

        Args:
            rv: The random variable to remove.

        Raises:
            KeyError: If the random variable is not in the dataset.
        """
        del self._data[rv]
        self._remove_rv(rv)

    def add_rv_from_state_weights(
            self,
            rv: RandomVariable,
            state_weights: NDArrayNumeric | Sequence[Sequence[float]],
    ) -> NDArrayNumeric:
        """
        Add a random variable to the dataset.

        When `state_weights` is a numpy array, then the dataset will directly reference the given array.

        Args:
            rv: The random variable to add.
            state_weights: A 2D array of state weights, with shape = `(len(self), len(rv))`.

        Raises:
            ValueError: If the random variable is already in the dataset.
        """
        if rv in self._data.keys():
            raise ValueError(f'data for {rv} already exists in the dataset')

        if not isinstance(state_weights, np.ndarray):
            state_weights = np.array(state_weights, dtype=np.float64)

        expected_shape = (self._length, len(rv))
        if state_weights.shape == expected_shape:
            rv_data = state_weights
        else:
            raise ValueError(f'data for {rv} expected shape {expected_shape}, got {state_weights.shape}')

        self._data[rv] = rv_data
        self._add_rv(rv)
        return rv_data

    def add_rv_from_state_idxs(self, rv: RandomVariable, state_idxs: NDArrayStates | Sequence[int]) -> NDArrayNumeric:
        """
        Add a random variable to the dataset.

        The dataset will directly reference the given `states` array.

        Args:
            rv: The random variable to add.
            state_idxs: An 1D array of state indexes to add, with shape = `(len(self),)`.
                Each element `state` should be `0 <= state < len(rv)`.

        Raises:
            ValueError: If the random variable is already in the dataset.
        """
        rv_data = np.zeros((len(state_idxs), len(rv)), dtype=np.float64)
        for i, state_idx in enumerate(state_idxs):
            rv_data[i, state_idx] = 1

        return self.add_rv_from_state_weights(rv, rv_data)

    def add_rv_from_states(self, rv: RandomVariable, states: Sequence[State]) -> NDArrayNumeric:
        """
        Add a random variable to the dataset.

        The dataset will allocate and populate a states array containing state indexes.
        This will call `rv.state_idx(state)` for each state in `states`.

        Args:
            rv: The random variable to add.
            states: An 1D array of state to add, with `len(states)` = `len(self)`.
                Each element `state` should be in `rv.states`.

        Raises:
            ValueError: If the random variable is already in the dataset.
        """
        rv_data = np.zeros((len(states), len(rv)), dtype=np.float64)
        for i, state in enumerate(states):
            state_idx = rv.state_idx(state)
            rv_data[i, state_idx] = 1

        return self.add_rv_from_state_weights(rv, rv_data)

    def soft_instances(
            self,
            rvs: Optional[Sequence[RandomVariable]] = None,
    ) -> Iterator[Tuple[Tuple[NDArrayNumeric], float]]:
        """
        Iterate over weighted instances  of soft evidence.

        Args:
            rvs: The random variables to include in iteration. Default is all dataset random variables.

        Returns:
            an iterator over (instance, weight) pairs, in the same order and number of instances in this dataset.
            An instance is a sequence of soft weights, co-indexed with `self.rvs`.
        """
        if rvs is None:
            rvs = self.rvs
        # Special case - no random variables
        if len(rvs) == 0:
            return zip(repeat(()), self.weights)
        else:
            cols: List[NDArrayNumeric] = [self.state_weights(rv) for rv in rvs]
            return zip(zip(*cols), self.weights)

    def hard_instances(self, rvs: Optional[Sequence[RandomVariable]] = None) -> Iterator[Tuple[Instance, float]]:
        """
        Iterate over equivalent weighted hard instances.

        Args:
            rvs: The random variables to include in iteration. Default is all dataset random variables.

        Returns:
            an iterator over (instance, weight) pairs where the order and number of instances
            is not guaranteed.
            An instance is a sequence of state indexes, co-indexed with `self.rvs`.
        """
        if rvs is None:
            rvs = self.rvs
        # Special case - no random variables
        if len(rvs) == 0:
            yield (), self.total_weight()
        else:
            for instance_weights, weight in self.soft_instances(rvs):
                if weight != 0:
                    for instance, instance_weight in _product_instance_weights(instance_weights):
                        yield instance, instance_weight * weight

    def dump(self, *, show_rvs: bool = True, show_weights: bool = True) -> None:
        """
        Dump the dataset in a human-readable format.

        Args:
            show_rvs: If `True`, the random variables are dumped.
            show_weights: If `True`, the instance weights are dumped.
        """
        if show_rvs:
            rvs = ', '.join(str(rv) for rv in self.rvs)
            print(f'rvs: [{rvs}]')
        print(f'instances ({len(self)}, with total weight {self.total_weight()}):')
        for instance, weight in self.soft_instances():
            instance_str = ', '.join(str(state_weights) for state_weights in instance)
            if show_weights:
                print(f'({instance_str}) * {weight}')
            else:
                print(f'({instance_str})')


def _product_instance_weights(instance_weights: Sequence[NDArrayNumeric]) -> Iterator[Tuple[Tuple[int, ...], float]]:
    """
    Iterate over all possible hard instances for the given
    instance weights, where the weight is not zero.

    This is a support function for `SoftDataset.hard_instances`.
    """

    # Base case
    if len(instance_weights) == 0:
        yield (), 1

    # Recursive case
    else:
        next_weights: NDArrayNumeric = instance_weights[-1]
        pre_weights: Sequence[NDArrayNumeric] = instance_weights[:-1]
        weight: float
        for pre_instance, pre_weight in _product_instance_weights(pre_weights):
            for i, weight in enumerate(next_weights):
                if weight != 0:
                    yield pre_instance + (int(i),), pre_weight * weight
