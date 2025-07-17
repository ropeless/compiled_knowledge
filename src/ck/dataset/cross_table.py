from typing import List, Tuple, Sequence, Iterator, Iterable, Optional, MutableMapping, Dict

from ck.dataset import SoftDataset, HardDataset
from ck.pgm import RandomVariable, rv_instances, Instance
from ck.utils.np_extras import NDArray


class CrossTable(MutableMapping[Instance, float]):
    """
    A cross-table records the total weight for possible combinations
    of states for some random variables, i.e., the weight of unique instances.

    A cross-table is a dictionary mapping from state indices of the cross-table
    random variables (an instance, as a tuple) to a weight (as a float).

    Given a cross-table `ct`, then for each `instance in ct.keys()`:
        `len(instance) == len(ct.rvs)`,
        and `0 <= instance[j] < len(ct.rvs[i])`,
        and `0 < ct[instance]`.

    Zero weighted instances are not explicitly represented in a cross-table.
    """

    def __init__(
            self,
            rvs: Sequence[RandomVariable],
            dirichlet_prior: float = 0,
            update: Iterable[Tuple[Instance, float]] = (),
    ):
        """
        Construct a cross-table for the given random variables.

        The cross-table can be initialised with a Dirichlet prior, x. Practically
        this amounts to adding a weight of x to each possible combination of
        random variable states. That is, a Dirichlet prior of x results in x pseudocounts
        for each possible combination of states.

        Args:
            rvs: the random variables that this cross-table records weights for. Instances
                in this cross-table are tuples of state indexes, co-indexed with `rvs`.
            dirichlet_prior: a real number >= 0, representing a Dirichlet prior.
            update: an optional iterable of (instance, weight) tuples to add to
                the cross-table at construction time.
        """
        self._rvs: Tuple[RandomVariable, ...] = tuple(rvs)
        self._dict: Dict[Instance, float]

        if dirichlet_prior != 0:
            instance: Tuple[int, ...]
            self._dict = {
                instance: dirichlet_prior
                for instance in rv_instances(*self._rvs)
            }
        else:
            self._dict = {}

        for instance, weight in update:
            self.add(instance, weight)

    def __eq__(self, other) -> bool:
        """
        Two cross-tables are equal if they have the same sequence of random variables
        and their instance weights are equal.
        """
        return isinstance(other, CrossTable) and self._rvs == other._rvs and self._dict == other._dict

    def __setitem__(self, key: Instance, value) -> None:
        if value == 0:
            self._dict.pop(key)
        else:
            self._dict[key] = value

    def __delitem__(self, key: Instance) -> None:
        del self._dict[key]

    def __getitem__(self, key: Instance) -> float:
        """
        Returns:
            the weight of the given instance.
            This will always return a value, even if the key is not in the underlying dictionary.
        """
        return self._dict.get(key, 0)

    def __len__(self) -> int:
        """
        Returns:
            the number of instances in the cross-table with non-zero weight.
        """
        return len(self._dict)

    def __iter__(self) -> Iterator[Instance]:
        """
        Returns:
            an iterator over the cross-table instances with non-zero weight.
        """
        return iter(self._dict)

    def items(self) -> Iterable[Tuple[Instance, float]]:
        """
        Returns:
            an iterable over (instance, weight) pairs.
        """
        return self._dict.items()

    @property
    def rvs(self) -> Sequence[RandomVariable]:
        """
        The random variables that this cross-table refers to.
        """
        return self._rvs

    def add(self, instance: Instance, weight: float) -> None:
        """
        Add the given weighted instance to the cross-table.

        Args:
            instance: a tuple of state indices, co-indexed with `self.rvs`.
            weight: the weight (generalised count) to add to the cross-table. Normally the
                weight will be > 0.
        """
        self[instance] = self._dict.get(instance, 0) + weight

    def total_weight(self) -> float:
        """
        Calculate the total weight of this cross-table.
        """
        return sum(self.values())


def cross_table_from_dataset(
        dataset: HardDataset | SoftDataset,
        rvs: Optional[Sequence[RandomVariable]] = None,
        *,
        dirichlet_prior: float = 0,
) -> CrossTable:
    """
    Generate a cross-table for the given random variables, using the given dataset, represented
    as a dictionary, mapping instances to weights.

    Args:
        dataset: The dataset to use to compute the cross-table.
        rvs: The random variables to compute the cross-table for. If omitted
            then `dataset.rvs` will be used.
        dirichlet_prior: a real number >= 0. See `CrossTable` for an explanation.

    Returns:
        The cross-table for the given random variables, using the given dataset,
        represented as a dictionary mapping instances to weights.
        An instance is a tuple of state indexes, co-indexed with rvs.

    Raises:
        KeyError: If any random variable in `rvs` does not appear in the dataset.
    """
    if isinstance(dataset, SoftDataset):
        return cross_table_from_soft_dataset(dataset, rvs, dirichlet_prior=dirichlet_prior)
    if isinstance(dataset, HardDataset):
        return cross_table_from_hard_dataset(dataset, rvs, dirichlet_prior=dirichlet_prior)
    raise TypeError('dataset must be either a SoftDataset or HardDataset')


def cross_table_from_soft_dataset(
        dataset: SoftDataset,
        rvs: Optional[Sequence[RandomVariable]] = None,
        *,
        dirichlet_prior: float = 0
) -> CrossTable:
    """
    Generate a cross-table for the given random variables, using the given dataset, represented
    as a dictionary, mapping instances to weights.

    Args:
        dataset: The dataset to use to compute the cross-table.
        rvs: The random variables to compute the cross-table for. If omitted
            then `dataset.rvs` will be used.
        dirichlet_prior: a real number >= 0. See `CrossTable` for an explanation.

    Returns:
        The cross-table for the given random variables, using the given dataset,
        represented as a dictionary mapping instances to weights.
        An instance is a tuple of state indexes, co-indexed with rvs.

    Raises:
        KeyError: If any random variable in `rvs` does not appear in the dataset.
    """
    if rvs is None:
        rvs = dataset.rvs

    # Special case
    if len(rvs) == 0:
        return CrossTable((), 0, [((), dataset.total_weight() + dirichlet_prior)])

    weights: CrossTable = CrossTable(rvs, dirichlet_prior)

    columns: List[NDArray] = [
        dataset.state_weights(rv)
        for rv in rvs
    ]

    for instance_weights, weight in zip(zip(*columns), dataset.weights):
        if weight != 0:
            for instance, instance_weight in _product_instance_weights(instance_weights):
                weights.add(instance, instance_weight * weight)

    return weights


def cross_table_from_hard_dataset(
        dataset: HardDataset,
        rvs: Optional[Sequence[RandomVariable]] = None,
        *,
        dirichlet_prior: float = 0
) -> CrossTable:
    """
    Generate a cross-table for the given random variables, using the given dataset, represented
    as a dictionary, mapping instances to weights.

    Args:
        dataset: The dataset to use to compute the cross-table.
        rvs: The random variables to compute the cross-table for. If omitted
            then `dataset.rvs` will be used.
        dirichlet_prior: a real number >= 0. See `CrossTable` for an explanation.

    Returns:
        The cross-table for the given random variables, using the given dataset,
        represented as a dictionary mapping instances to weights.
        An instance is a tuple of state indexes, co-indexed with rvs.

    Raises:
        KeyError: If any random variable in `rvs` does not appear in the dataset.
    """
    if rvs is None:
        rvs = dataset.rvs

    # Special case
    if len(rvs) == 0:
        return CrossTable((), 0, [((), dataset.total_weight() + dirichlet_prior)])

    weights: CrossTable = CrossTable(rvs, dirichlet_prior)

    columns: List[NDArray] = [
        dataset.state_idxs(rv)
        for rv in rvs
    ]

    for instance, weight in zip(zip(*columns), dataset.weights):
        if weight != 0:
            instance: Tuple[int, ...] = tuple(int(i) for i in instance)
            weights.add(instance, weight)

    return weights


def _product_instance_weights(instance_weights: Sequence[NDArray]) -> Iterator[Tuple[Tuple[int, ...], float]]:
    """
    Iterate over all possible instance for the given instance weights,
    where the weight is not zero.
    """

    # Base case
    if len(instance_weights) == 0:
        yield (), 1

    # Recursive case
    else:
        next_weights: NDArray = instance_weights[-1]
        pre_weights: Sequence[NDArray] = instance_weights[:-1]
        for pre_instance, pre_weight in _product_instance_weights(pre_weights):
            for i, weight in enumerate(next_weights):
                if weight != 0:
                    yield pre_instance + (int(i),), pre_weight * weight
