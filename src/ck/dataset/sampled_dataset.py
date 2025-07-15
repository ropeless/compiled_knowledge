import random
from dataclasses import dataclass
from typing import Sequence, List, Iterator, Tuple, Dict

import numpy as np

from ck.dataset import HardDataset
from ck.dataset.cross_table import CrossTable
from ck.pgm import RandomVariable, Instance
from ck.sampling.sampler import Sampler
from ck.utils.np_extras import dtype_for_number_of_states, NDArray
from ck.utils.random_extras import Random


def dataset_from_sampler(sampler: Sampler, length: int) -> HardDataset:
    """
    Create a hard dataset using samples from a sampler.

    Args:
        sampler: A sampler which defined the random variables and provides samples.
        length: The length of the dataset to create.

    Returns:
        A HardDataset of the given length.
    """
    rvs: Sequence[RandomVariable] = sampler.rvs
    columns: List[NDArray] = [
        np.zeros(length, dtype=dtype_for_number_of_states(len(rv)))
        for rv in rvs
    ]
    for i, instance in enumerate(sampler.take(length)):
        for column, state in zip(columns, instance):
            column[i] = state
    return HardDataset(zip(rvs, columns))


class CrossTableSampler(Sampler):
    def __init__(self, crosstab: CrossTable, rand: Random = random):
        """
        Adapt a cross table to a sampler.

        Instances will be drawn from the sampler according to their
        weight in the given cross-table. If the given cross-table is
        modified after constructing the sampler, the sampler will not
        be affected.
        """
        if len(crosstab) == 0:
            raise ValueError('no instances to sample')

        super().__init__(rvs=crosstab.rvs, condition=())

        # Group instances by weight.
        # We do this in anticipation that it makes sampling more efficient.
        weight_groups: Dict[float, _WeightGroup] = {}
        for instance, weight in crosstab.items():
            weight_group = weight_groups.get(weight)
            if weight_group is None:
                weight_groups[weight] = _WeightGroup(weight, weight, [instance])
            else:
                weight_group.append(instance)

        self._weight_groups: List[_WeightGroup] = list(weight_groups.values())
        self._total_weight = sum(group.total for group in weight_groups.values())
        self._rand = rand

    def __iter__(self) -> Iterator[Instance]:
        while True:
            # This code performs inverse transform sampling
            r: float = self._rand.random() * self._total_weight

            # This does a serial search to find the weight group.
            # This is efficient for small numbers of groups, but this may be
            # improved for large numbers of groups.
            it = iter(self._weight_groups)
            group = next(it)
            while r >= group.total:
                r -= group.total
                group = next(it)

            # Pick an instance in the group
            i = int(r / group.weight)
            yield group.instances[i]


@dataclass
class _WeightGroup:
    """
    Support for CrossTableSampler.
    """
    weight: float
    total: float
    instances: List[Tuple[int, ...]]

    def append(self, instance: Tuple[int, ...]) -> None:
        self.total += self.weight
        self.instances.append(instance)
