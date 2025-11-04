from __future__ import annotations

from typing import List, Tuple, Sequence, Iterator, Iterable, Optional, MutableMapping, Dict, assert_never

from ck.dataset import SoftDataset, HardDataset
from ck.pgm import RandomVariable, rv_instances, Instance


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
    Given a cross-table `ct` then the following is always true.
    `x in ct.keys()` is true if and only if `ct[x] != 0`.
    """

    def __init__(
            self,
            rvs: Sequence[RandomVariable],
            dirichlet_prior: float | CrossTable = 0,
            update: Iterable[Tuple[Instance, float]] = (),
    ):
        """
        Construct a cross-table for the given random variables.

        The cross-table can be initialised with a Dirichlet prior, x. Practically
        this amounts to adding a weight of x to each possible combination of
        random variable states. That is, a Dirichlet prior of x results in x pseudocounts
        for each possible combination of states.

        To copy a cross-table, `x`, you can use the following idiom:
        ```
            x_copy = CrossTable(x.rvs, update=x.items())
        ```

        Args:
            rvs: the random variables that this cross-table records weights for. Instances
                in this cross-table are tuples of state indexes, co-indexed with `rvs`.
            dirichlet_prior: provides a prior for `rvs`. This can be represented either:
                (a) as a uniform prior, represented as a float value,
                (b) as an arbitrary prior, represented as a cross-table.
                If a cross-table is provided as a prior, then it must have the same random variables as `rvs`.
                The default value for `dirichlet_prior` is 0.
            update: an optional iterable of (instance, weight) tuples to add to
                the cross-table at construction time.
        """
        self._rvs: Tuple[RandomVariable, ...] = tuple(rvs)
        self._dict: Dict[Instance, float]

        if isinstance(dirichlet_prior, CrossTable):
            # rv_map[i] is where rvs[i] appears in the dirichlet_prior cross-table
            # It will be used to map instances of the prior to instances of self.
            rv_map: List[int] = [
                dirichlet_prior.rvs.index(rv)
                for rv in rvs
            ]

            # Copy items from the prior to self, mapping the instances as needed
            self._dict = {
                tuple(prior_instance[select] for select in rv_map): weight
                for prior_instance, weight in dirichlet_prior.items()
            }

        elif isinstance(dirichlet_prior, (float, int)):
            if dirichlet_prior != 0:
                # Initialise self with every possible combination of rvs states.
                instance: Instance
                self._dict = {
                    instance: dirichlet_prior
                    for instance in rv_instances(*self._rvs)
                }
            else:
                self._dict = {}
        else:
            assert_never('not reached')

        # Apply any provided updates
        self.add_all(update)

    def __eq__(self, other) -> bool:
        """
        Two cross-tables are equal if they have the same sequence of random variables
        and their instance weights are equal.
        """
        return isinstance(other, CrossTable) and self._rvs == other._rvs and self._dict == other._dict

    def __setitem__(self, key: Instance, value) -> None:
        if value == 0:
            self._dict.pop(key, None)
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
            an iterable over (instance, weight) pairs, where weight != 0.
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

    def add_all(self, to_add: Iterable[Tuple[Instance, float]]) -> None:
        """
        Add the given weighted instances to the cross-table.

        Args:
            to_add: an iterable of (instance, weight) tuples to add to the cross-table.
        """
        for instance, weight in to_add:
            self.add(instance, weight)

    def mul(self, multiplier: float) -> None:
        """
        Multiply all weights by the given multiplier.
        """
        if multiplier == 0:
            self._dict.clear()
        elif multiplier == 1:
            pass
        else:
            for instance in self._dict.keys():
                self._dict[instance] *= multiplier

    def total_weight(self) -> float:
        """
        Calculate the total weight of this cross-table.
        """
        return sum(self.values())

    def project(self, rvs: Sequence[RandomVariable]) -> CrossTable:
        """
        Project this cross-table onto the given set of random variables.

        If successful, this method will always return a new CrossTable object.

        Returns:
            a CrossTable with the given sequence of random variables.

        Assumes:
            `rvs` is a subset of the cross-table's random variables.
        """
        # Mapping rv_map[i] is the index into `self.rvs` for `rvs[i]`.
        rv_map: List[int] = [self.rvs.index(rv) for rv in rvs]

        return CrossTable(
            rvs=rvs,
            update=(
                (tuple(instance[i] for i in rv_map), weight)
                for instance, weight in self._dict.items()
            ),
        )

    def dump(self, *, show_rvs: bool = True, show_weights: bool = True, as_states: bool = False) -> None:
        """
        Dump the cross-table in a human-readable format.
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
        for instance, weight in self.items():
            if as_states:
                instance_str = ', '.join(repr(rv.states[idx]) for idx, rv in zip(instance, self.rvs))
            else:
                instance_str = ', '.join(str(idx) for idx in instance)
            if show_weights:
                print(f'({instance_str}) * {weight}')
            else:
                print(f'({instance_str})')


def cross_table_from_dataset(
        dataset: HardDataset | SoftDataset,
        rvs: Optional[Sequence[RandomVariable]] = None,
        *,
        dirichlet_prior: float | CrossTable = 0,
) -> CrossTable:
    """
    Generate a cross-table for the given random variables, using the given dataset, represented
    as a dictionary, mapping instances to weights.

    Args:
        dataset: The dataset to use to compute the cross-table.
        rvs: The random variables to compute the cross-table for. If omitted
            then `dataset.rvs` will be used.
        dirichlet_prior: provides a Dirichlet prior for `rvs`. This can be represented either:
            (a) as a uniform prior, represented as a float value,
            (b) as an arbitrary Dirichlet prior, represented as a cross-table.
            If a cross-table is provided as a prior, then it must have the same random variables as `rvs`.
            The default value for `dirichlet_prior` is 0.
            See `CrossTable` for more explanation.

    Returns:
        The cross-table for the given random variables, using the given dataset,
        represented as a dictionary mapping instances to weights.
        An instance is a tuple of state indexes, co-indexed with rvs.

    Raises:
        KeyError: If any random variable in `rvs` does not appear in the dataset.
    """
    if isinstance(dataset, HardDataset):
        return cross_table_from_hard_dataset(dataset, rvs, dirichlet_prior=dirichlet_prior)
    if isinstance(dataset, SoftDataset):
        return cross_table_from_soft_dataset(dataset, rvs, dirichlet_prior=dirichlet_prior)
    raise TypeError('dataset must be either a SoftDataset or HardDataset')


def cross_table_from_hard_dataset(
        dataset: HardDataset,
        rvs: Optional[Sequence[RandomVariable]] = None,
        *,
        dirichlet_prior: float | CrossTable = 0
) -> CrossTable:
    """
    Generate a cross-table for the given random variables, using the given dataset, represented
    as a dictionary, mapping instances to weights.

    Args:
        dataset: The dataset to use to compute the cross-table.
        rvs: The random variables to compute the cross-table for. If omitted
            then `dataset.rvs` will be used.
        dirichlet_prior: provides a Dirichlet prior for `rvs`. This can be represented either:
            (a) as a uniform prior, represented as a float value,
            (b) as an arbitrary Dirichlet prior, represented as a cross-table.
            If a cross-table is provided as a prior, then it must have the same random variables as `rvs`.
            The default value for `dirichlet_prior` is 0.
            See `CrossTable` for more explanation.

    Returns:
        The cross-table for the given random variables, using the given dataset,
        represented as a dictionary mapping instances to weights.
        An instance is a tuple of state indexes, co-indexed with rvs.

    Raises:
        KeyError: If any random variable in `rvs` does not appear in the dataset.
    """
    if rvs is None:
        rvs = dataset.rvs
    return CrossTable(
        rvs=rvs,
        dirichlet_prior=dirichlet_prior,
        update=dataset.instances(rvs)
    )


def cross_table_from_soft_dataset(
        dataset: SoftDataset,
        rvs: Optional[Sequence[RandomVariable]] = None,
        *,
        dirichlet_prior: float | CrossTable = 0
) -> CrossTable:
    """
    Generate a cross-table for the given random variables, using the given dataset, represented
    as a dictionary, mapping instances to weights.

    Args:
        dataset: The dataset to use to compute the cross-table.
        rvs: The random variables to compute the cross-table for. If omitted
            then `dataset.rvs` will be used.
        dirichlet_prior: provides a Dirichlet prior for `rvs`. This can be represented either:
            (a) as a uniform prior, represented as a float value,
            (b) as an arbitrary Dirichlet prior, represented as a cross-table.
            If a cross-table is provided as a prior, then it must have the same random variables as `rvs`.
            The default value for `dirichlet_prior` is 0.
            See `CrossTable` for more explanation.

    Returns:
        The cross-table for the given random variables, using the given dataset,
        represented as a dictionary mapping instances to weights.
        An instance is a tuple of state indexes, co-indexed with rvs.

    Raises:
        KeyError: If any random variable in `rvs` does not appear in the dataset.
    """
    if rvs is None:
        rvs = dataset.rvs

    return CrossTable(
        rvs=rvs,
        dirichlet_prior=dirichlet_prior,
        update=dataset.hard_instances(rvs)
    )
