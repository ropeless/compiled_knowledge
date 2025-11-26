from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, List, Tuple, Set, Iterable

from ck.dataset.cross_table import CrossTable
from ck.pgm import RandomVariable, Instance
from ck.utils.map_list import MapList


def condition(
        conditioned: CrossTable,
        conditioner: CrossTable,
        condition_rvs: Optional[Iterable[RandomVariable]] = None,
) -> CrossTable:
    """
    Apply the conditioner cross-table to the conditioned cross-table.

    Args:
        conditioned: the cross-table to condition.
        conditioner: the cross-table providing a distribution over condition random variables.
        condition_rvs: optional set of condition random variables. If not provided, then the intersection
            of `conditioned.rvs` and `conditioner.rvs` is used.

    Assumes:
        `condition_rvs_set` is a subset of `conditioned.rvs` and `conditioner.rvs`.

    Ensures:
        for every (instance, weight) in conditioner.project(rvs):
        conditioned.project(rvs)[instance] == weight,
        where rvs = list(condition_rvs).
    """
    if condition_rvs is None:
        # Infer the default condition random variables
        condition_rvs_set = set(conditioned.rvs).intersection(conditioner.rvs)
    elif isinstance(condition_rvs, (set, frozenset)) is None:
        condition_rvs_set = condition_rvs
    else:
        condition_rvs_set = set(condition_rvs)

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
