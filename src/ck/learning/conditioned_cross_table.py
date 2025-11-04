from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, List, Tuple, Set, Iterable

from ck.dataset.cross_table import CrossTable
from ck.pgm import RandomVariable, Instance
from ck.utils.map_list import MapList


@dataclass
class ConditionedCrossTable:
    """
    A conditioned cross-table is a cross-table where the distribution over some
    random variables is not reliable and should be adjusted according to some
    other cross-table.
    """
    cross_table: CrossTable
    condition_rvs: Sequence[RandomVariable]  # must be a subset of `cross_table.rvs`

    def __post_init__(self):
        # ensure `condition_rvs` is a subset of `cross_table.rvs`
        if not set(self.condition_rvs).issubset(self.cross_table.rvs):
            raise ValueError('condition_rvs is not a subset of cross_table.rvs')

    @property
    def is_unconditioned(self) -> bool:
        """
        The cross-table is unconditioned if there are no condition random variables.
        """
        return len(self.condition_rvs) == 0

    def condition(
            self,
            conditioner: CrossTable,
            condition_rvs: Optional[Sequence[RandomVariable]] = None,
    ) -> ConditionedCrossTable:
        """
        Apply the given cross-table to remove condition random variables.

        Args:
            conditioner: the cross-table providing a distribution over condition random variables.
            condition_rvs: the condition random variables to remove. If not provided, then
                the intersection of `self.cross_table.rvs` and `conditioner.rvs` is used.`. If provided,
                then they must be a subset of `self.cross_table.rvs` and `conditioner.rvs`. If
                `condition_rvs` is empty, then the result is just a copy of `self`.
        """
        condition_rvs_set: Set[RandomVariable]
        if condition_rvs is None:
            # Infer the default condition random variables
            condition_rvs_set = set(self.cross_table.rvs).intersection(conditioner.rvs)
        else:
            condition_rvs_set = set(condition_rvs)

        new_cross_table = condition(
            conditioned=self.cross_table,
            conditioner=conditioner,
            condition_rvs_set=condition_rvs_set,
        )

        # Get the remaining condition_rvs
        new_condition_rvs = tuple(rv for rv in self.condition_rvs if rv not in condition_rvs_set)

        # Bring the result together
        return ConditionedCrossTable(
            cross_table=new_cross_table,
            condition_rvs=new_condition_rvs,
        )


def condition(
        conditioned: CrossTable,
        conditioner: CrossTable,
        condition_rvs_set: Set[RandomVariable],
) -> CrossTable:
    """
    Apply the conditioner cross-table to the conditioned cross-table.

    Args:
        conditioned: the cross-table to condition.
        conditioner: the cross-table providing a distribution over condition random variables.
        condition_rvs_set: the condition random variables.

    Assumes:
        `condition_rvs_set` is a subset of `conditioned.rvs` and `conditioner.rvs`.

    Ensures:
        for every instance, weight in conditioner.project(rvs):
            conditioned.project(rvs)[instance] == weight
        where rvs = list(condition_rvs_set).
    """
    # Stabilize the list of condition random variables
    condition_rvs = tuple(condition_rvs_set)

    if condition_rvs_set != set(conditioner.rvs):
        # Sum out irrelevant random variables from `conditioner`
        conditioner = conditioner.project(condition_rvs)
    else:
        # Ensure `condition_rvs` is the same order as conditioner.rvs
        condition_rvs = conditioner.rvs
    assert condition_rvs == conditioner.rvs, 'expect identical tuples'

    # Mapping rv_map[i] is the index into `cross_table.rvs` for `condition_rvs[i]`.
    rv_map: List[int] = [conditioned.rvs.index(rv) for rv in condition_rvs]

    # Group the cross-table instances by the condition_rvs
    groups: MapList[Instance, Tuple[Instance, float]] = MapList()
    for instance_weight in conditioned.items():
        instance, _ = instance_weight
        group = tuple(instance[i] for i in rv_map)
        groups.append(group, instance_weight)

    # Reweight each group
    def _reweighted_instances() -> Iterable[Tuple[Instance, float]]:
        nonlocal groups
        for _group, _instance_weights in groups.items():
            new_weight: float = conditioner[_group]
            if new_weight != 0:
                group_weight: float = sum(w for _, w in _instance_weights)
                multiplier: float = new_weight / group_weight
                for _instance, _weight in _instance_weights:
                    yield _instance, _weight * multiplier

    return CrossTable(
        rvs=conditioned.rvs,
        update=_reweighted_instances(),
    )
