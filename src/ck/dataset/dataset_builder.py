from __future__ import annotations

from itertools import count
from typing import Iterable, List, TypeAlias, Sequence, overload, Set, Tuple, MutableSequence, Dict, Optional, \
    assert_never

import numpy as np

from ck.dataset import HardDataset, SoftDataset
from ck.pgm import RandomVariable, State
from ck.utils.np_extras import NDArrayFloat64, NDArrayStates, dtype_for_number_of_states, DTypeStates, NDArrayNumeric

HardValue: TypeAlias = int
SoftValue: TypeAlias = Sequence[float]
Value: TypeAlias = HardValue | SoftValue | None


class Record(Sequence[Value]):
    """
    A record is a sequence of values, co-indexed with dataset columns.

    A value is either a state index (HardValue), a sequence of state
    weights (SoftValue), or missing (None).
    """

    def __init__(self, dataset: DatasetBuilder, values: Optional[Iterable[Value]] = None):
        self.weight: float = 1
        self._dataset: DatasetBuilder = dataset
        self._values: List[Value] = [] if values is None else list(values)

    def __len__(self) -> int:
        return len(self._dataset.rvs)

    @overload
    def __getitem__(self, index: int | RandomVariable) -> Value:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Value]:
        ...

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._getitem(i) for i in range(*index.indices(len(self)))]
        if isinstance(index, RandomVariable):
            # noinspection PyProtectedMember
            return self._getitem(self._dataset._rvs_index[index])

        size = len(self)
        if index < 0:
            index += size
        if not 0 <= index < size:
            raise IndexError('index out of range')
        return self._getitem(index)

    def _getitem(self, index: int) -> Value:
        """
        Assumes:
            0 <= index < len(self).
        """
        if index >= len(self._values):
            return None
        return self._values[index]

    @overload
    def __setitem__(self, index: int | RandomVariable, value: Value) -> None:
        ...

    @overload
    def __setitem__(self, index: slice, value: Iterable[Value]) -> None:
        ...

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            for i, v in zip(range(*index.indices(len(self))), value):
                self._setitem(i, v)
            return
        if isinstance(index, RandomVariable):
            # noinspection PyProtectedMember
            self._setitem(self._dataset._rvs_index[index], value)
            return

        size = len(self)
        if index < 0:
            index += size
        if not 0 <= index < size:
            raise IndexError('index out of range')
        self._setitem(index, value)

    def _setitem(self, index: int, value: Value) -> None:
        """
        Assumes:
            0 <= index < len(self).
        """
        to_append: int = index + 1 - len(self._values)
        self._values += [None] * to_append

        if value is None:
            self._values[index] = None
            return

        rv: RandomVariable = self._dataset.rvs[index]
        if isinstance(value, int):
            if not (0 <= value < len(rv)):
                raise ValueError(f'state index out of range, expected: 0 <= {value!r} < {len(rv)}')
            self._values[index] = value
            return

        # Expect the value is a sequence of floats
        if len(value) != len(rv):
            raise ValueError(f'state weights incorrect length, expected: {len(rv)}, got: {len(value)}')
        self._values[index] = tuple(value)

    def set(self, *values: Value) -> None:
        """
        Set all the values of this record, using state indexes or state weights.

        If insufficient or additional values are provided, a ValueError will be raised.
        """
        if len(values) != len(self):
            raise ValueError('incorrect number of values provided')
        for i, value in enumerate(values):
            self._setitem(i, value)

    def set_states(self, *values: State) -> None:
        """
        Set all the values of this record from random variable states.

        State indexes are resolved using `RandomVariable.state_idx`.
        If insufficient or additional values are provided, a ValueError will be raised.
        """
        rvs = self._dataset.rvs
        if len(values) != len(rvs):
            raise ValueError('incorrect number of values provided')
        for i, rv, value in zip(count(), rvs, values):
            self._setitem(i, rv.state_idx(value))

    def __str__(self) -> str:
        return self.to_str()

    def to_str(
            self,
            *,
            show_weight: bool = True,
            as_states: bool = False,
            missing: str = 'None',
            sep: str = ', ',
    ) -> str:
        """
        Render the record as a human-readable string.
        If as_states is true, then hard values states are dumped instead of just state indexes.

        Args:
            show_weight: If `True`, the instance weight is included.
            as_states: If `True`, the states are used instead of just state indexes.
            missing: the string to use for missing values.
            sep: the string to use for separating values.
        """

        def _value_str(rv_idx: int, v: Value) -> str:
            if v is None:
                return missing
            if isinstance(v, int):
                if as_states:
                    return repr(self._dataset.rvs[rv_idx].states[v])
                else:
                    return str(v)
            else:
                return str(v)

        instance_str = sep.join(_value_str(i, self._getitem(i)) for i in range(len(self)))
        if show_weight:
            return f'({instance_str}) * {self.weight}'
        else:
            return f'({instance_str})'


class DatasetBuilder(Sequence[Record]):
    """
    A dataset builder can be used for making a hard or soft dataset, incrementally growing
    the dataset as needed. This represents a flexible but inefficient interim representation of data.
    """

    def __init__(self, rvs: Iterable[RandomVariable] = ()):
        """
        Args:
            rvs: Optional random variables to include in the dataset. Default is no random variables.
        """
        self._rvs: Tuple[RandomVariable, ...] = ()
        self._rvs_index: Dict[RandomVariable, int] = {}
        self._records: List[Record] = []
        self.new_column(*rvs)

    @property
    def rvs(self) -> Sequence[RandomVariable]:
        return self._rvs

    def new_column(self, *rv: RandomVariable) -> None:
        """
        Adds one, or more, new random variables to the dataset. For existing rows,
        value for the new random variable will be `None`.

        Args:
            rv: a new random variable to include in the dataset.

        Raises:
            ValueError: if the given random variable already exists in the dataset.
        """
        # Do all consistency checks first to fail early, before modifying the dataset.
        rvs_to_add: Set[RandomVariable] = set(rv)
        if len(rvs_to_add) != len(rv):
            raise ValueError(f'request to add a column includes duplicates')
        duplicate_rvs: Set[RandomVariable] = rvs_to_add.intersection(self._rvs_index.keys())
        if len(duplicate_rvs) > 0:
            duplicate_rv_names = ', '.join(rv.name for rv in duplicate_rvs)
            raise ValueError(f'column already exists in the dataset: {duplicate_rv_names}')

        for rv in rvs_to_add:
            self._rvs_index[rv] = len(self._rvs)
            self._rvs += (rv,)

    def ensure_column(self, *rv: RandomVariable) -> None:
        """
        Add a column for one, or more, random variables, only
        adding a random variable if it is not already present in the dataset.
        """
        all_rvs = self._rvs_index.keys()
        self.new_column(*(_rv for _rv in rv if _rv not in all_rvs))

    def del_column(self, *rv: RandomVariable) -> None:
        """
        Delete one, or more, random variables from the dataset.

        Args:
            rv: a random variable to remove from the dataset.

        Raises:
            ValueError: if the given random variable does not exist in the dataset.
        """
        # Do all consistency checks first to fail early, before modifying the dataset.
        rvs_to_del: Set[RandomVariable] = set(rv)
        if len(rvs_to_del) != len(rv):
            raise ValueError(f'request to delete a column includes duplicates')
        missing_columns = rvs_to_del.difference(self._rvs_index.keys())
        if len(missing_columns) > 0:
            missing_rv_names = ', '.join(rv.name for rv in missing_columns)
            raise ValueError(f'missing columns: {missing_rv_names}')

        # Get column indices to remove, in descending order
        indices = sorted((self._rvs_index[rv] for rv in rvs_to_del), reverse=True)

        # Remove from the index
        for rv in rvs_to_del:
            self._rvs_index.pop(rv)

        # Remove from column sequence
        rvs_list: List[RandomVariable] = list(self._rvs)
        for i in indices:
            rvs_list.pop(i)
        self._rvs = tuple(rvs_list)

        # Remove from records
        for record in self._records:
            # noinspection PyProtectedMember
            record_values: List[Value] = record._values
            for i in indices:
                if i < len(record_values):
                    record_values.pop(i)

    def total_weight(self) -> float:
        """
        Calculate the total weight of this dataset.
        """
        return sum(record.weight for record in self._records)

    def get_weights(self) -> NDArrayFloat64:
        """
        Allocate and return a 1D numpy array of instance weights.

        Ensures:
            shape of the result == `(len(self), )`.
        """
        result: NDArrayStates = np.fromiter(
            (record.weight for record in self._records),
            count=len(self._records),
            dtype=np.float64,
        )
        return result

    def get_column_hard(self, rv: RandomVariable, *, missing: Optional[int] = None) -> NDArrayStates:
        """
        Allocate and return a 1D numpy array of state indexes.

        The state of a random variable (for an instance) where the value is soft evidence,
        is the state with the maximum weight. Ties are broken arbitrarily.

        Args:
            rv: a random variable in this dataset.
            missing: the value to use in the result to represent missing values. If not provided,
                then the default missing value is len(rv), which is an invalid state index.

        Raises:
            ValueError: if the supplied missing value is negative.

        Ensures:
            shape of the result == `(len(self), )`.
        """
        index: int = self._rvs_index[rv]
        if missing is None:
            missing = len(rv)
        if missing < 0:
            raise ValueError(f'missing value must be >= 0')
        number_of_states = max(len(rv), missing + 1)
        dtype: DTypeStates = dtype_for_number_of_states(number_of_states)
        result: NDArrayStates = np.fromiter(
            (_get_state(record[index], missing) for record in self._records),
            count=len(self._records),
            dtype=dtype,
        )
        return result

    def get_column_soft(self, rv: RandomVariable, *, missing: float | Sequence[float] = np.nan) -> NDArrayFloat64:
        """
        Allocate and return a numpy array of state weights.

        Args:
            rv: a random variable in this dataset.
            missing: the value to use in the result to represent missing values. Default is all NaN.

        Ensures:
            shape of the result == `(len(self), len(rv))`.
        """
        index: int = self._rvs_index[rv]
        size: int = len(rv)

        if isinstance(missing, (float, int)):
            missing_weights: NDArrayFloat64 = np.array([missing] * size, dtype=np.float64)
        else:
            missing_weights: NDArrayFloat64 = np.array(missing, dtype=np.float64)
            if missing_weights.shape != (size,):
                raise ValueError(f'missing weights shape expected {(size,)}, but got {missing_weights.shape}')

        result: NDArrayFloat64 = np.empty(shape=(len(self._records), size), dtype=np.float64)
        for i, record in enumerate(self._records):
            result[i, :] = _get_state_weights(size, record[index], missing_weights)
        return result

    def append(self, *values: Value) -> Record:
        """
        Appends a new record to the dataset.

        Args:
            values: the new record to append. If omitted, a new record will be created
                with all values missing (`None`).

        Returns:
            the new record.
        """
        record = Record(self, values)
        self._records.append(record)
        return record

    def insert(self, index: int, values: Optional[Iterable[Value]] = None) -> Record:
        """
        Inserts a new record to the dataset at the given index.

        Args:
            index: where to insert the record (interpreted as per builtin `list.insert`).
            values: the new record to append. If omitted, a new record will be created
                with all values missing (`None`).

        Returns:
            the new record.
        """
        record = Record(self, values)
        self._records.insert(index, record)
        return record

    def append_dataset(self, dataset: HardDataset | SoftDataset) -> None:
        """
        Append all the records of the given dataset to this dataset builder.

        Args:
            dataset: the dataset of records to append.

        Raises:
            KeyError: if `dataset.rvs` is not a superset of `this.rvs` and ensure_cols is false.
                If you want to avoid this error, first call `self.ensure_column(*dataset.rvs)`.
        """
        if isinstance(dataset, HardDataset):
            cols: Tuple = tuple(dataset.state_idxs(rv).tolist() for rv in self.rvs)
        elif isinstance(dataset, SoftDataset):
            cols: Tuple = tuple(dataset.state_weights(rv) for rv in self.rvs)
        else:
            assert_never('not reached')
        weights: NDArrayNumeric = dataset.weights
        for weight, vals in zip(weights, zip(*cols)):
            self.append(*vals).weight = weight

    @overload
    def __getitem__(self, index: int) -> Record:
        ...

    @overload
    def __getitem__(self, index: slice) -> MutableSequence[Record]:
        ...

    def __getitem__(self, index):
        return self._records[index]

    def __delitem__(self, index: int | slice) -> None:
        del self._records[index]

    def __len__(self) -> int:
        return len(self._records)

    def dump(
            self,
            *,
            show_rvs: bool = True,
            show_weights: bool = True,
            as_states: bool = False,
            missing: str = 'None',
            sep: str = ', ',
    ) -> None:
        """
        Dump the dataset in a human-readable format.
        If as_states is true, then hard values states are dumped instead of just state indexes.

        Args:
            show_rvs: If `True`, the random variables are dumped.
            show_weights: If `True`, the instance weights are dumped.
            as_states: If `True`, the states are dumped instead of just state indexes.
            missing: the string to use for missing values.
            sep: the string to use for separating values.
        """
        if show_rvs:
            rvs = ', '.join(str(rv) for rv in self.rvs)
            print(f'rvs: [{rvs}]')
        print(f'instances ({len(self)}, with total weight {self.total_weight()}):')
        for record in self._records:
            print(record.to_str(show_weight=show_weights, as_states=as_states, missing=missing, sep=sep))


def hard_dataset_from_builder(dataset_builder: DatasetBuilder, *, missing: Optional[int] = None) -> HardDataset:
    """
    Create a hard dataset from a soft dataset by repeated application
    of `HardDataset.add_rv_from_state_idxs` using values from `self.get_column_hard`.

    The state of a random variable (for an instance) where the value is soft evidence,
    is the state with the maximum weight. Ties are broken arbitrarily.

    The instance weights of the returned dataset will simply
    be the weights from the builder.

    No adjustments are made to the resulting dataset weights, even if
    a value in the dataset builder is soft evidence that does not sum to
    one.

    Args:
        dataset_builder: The dataset builder providing random variables,
            their states, and instance weights.
        missing: the value to use in the result to represent missing values. If not provided,
            then the default missing value is len(rv) for each rv, which is an invalid state index.

    Returns:
        A `HardDataset` instance.
    """
    dataset = HardDataset(weights=dataset_builder.get_weights())
    for rv in dataset_builder.rvs:
        dataset.add_rv_from_state_idxs(rv, dataset_builder.get_column_hard(rv, missing=missing))
    return dataset


def soft_dataset_from_builder(
        dataset_builder: DatasetBuilder,
        *,
        missing: float | Sequence[float] = np.nan,
) -> SoftDataset:
    """
    Create a soft dataset from a hard dataset by repeated application
    of `SoftDataset.add_rv_from_state_idxs`.

    The instance weights of the returned dataset will be a copy
    of the instance weights of the hard dataset.

    Args:
        dataset_builder: The dataset builder providing random variables,
            their state weights, and instance weights.
        missing: the value to use in the result to represent missing values.
            If a single float is provided, all state weights will have that value. Alternatively,
            a sequence of state weights can be provided, but all random variables will need
            to be the same size. Default is all state weights set to NaN.

    Returns:
        A `SoftDataset` instance.
    """
    dataset = SoftDataset(weights=dataset_builder.get_weights())
    for rv in dataset_builder.rvs:
        dataset.add_rv_from_state_weights(rv, dataset_builder.get_column_soft(rv, missing=missing))
    return dataset


def _get_state(value: Value, missing: int) -> int:
    if value is None:
        return missing
    if isinstance(value, int):
        return value
    return np.argmax(value).item()


def _get_state_weights(size: int, value: Value, missing: Sequence[float]) -> Sequence[float]:
    if value is None:
        return missing
    if isinstance(value, int):
        result = np.zeros(size, dtype=np.float64)
        result[value] = 1
        return result
    return value
