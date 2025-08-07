from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import chain
from typing import Iterable, List, Tuple, Dict, Sequence, Set, Optional

from ck.dataset.cross_table import CrossTable
from ck.learning.coalesce_cross_tables import coalesce_cross_tables
from ck.learning.parameters import make_factors, ParameterValues
from ck.learning.train_generative_bn import cpt_and_parent_sums_from_crosstab
from ck.pgm import PGM, RandomVariable
from ck.utils.map_list import MapList


class ParameterInference(Enum):
    """
    There are variations on the method for inferring a CPT's parameter values.
    This enum defines the possible variations.
    """
    first = auto()
    """
    Use the first cross-table found that covers the CPT's random variables.
    If there is no such cross-table, revert to the `all` method.
    This is the fastest method.
    """

    sum = auto()
    """
    Use the sum of cross-tables that cover the CPT's random variables.
    If there are no such cross-tables, revert to the `all` method.
    This is the second fastest method.
    """

    all = auto()
    """
    Project all cross-tables onto the needed random variables, then use
    `coalesce_cross_tables` to solve for the best parameter values.
    """


def model_from_cross_tables(
        pgm: PGM,
        cross_tables: Iterable[CrossTable],
        method: ParameterInference = ParameterInference.sum,
) -> None:
    """
    Make best efforts to construct a Bayesian network model given only the
    evidence from the supplied cross-tables.

    This function calls `get_cpts` which provides parameters to define
    a Bayesian network model. These are then applied to the PGM using
    `make_factors`.

    Args:
        pgm: the PGM to add factors and potential function to.
        cross_tables: available cross-tables to build a model from.
        method: what parameter inference method to use.

    Raises:
        ValueError: If `pgm` has any existing factors.
    """
    if len(pgm.factors) > 0:
        raise ValueError('the given PGM should have no factors')
    cpts: List[CrossTable] = get_cpts(
        rvs=pgm.rvs,
        cross_tables=cross_tables,
        method=method,
    )
    make_factors(pgm, cpts)


def get_cpts(
        rvs: Sequence[RandomVariable],
        cross_tables: Iterable[CrossTable],
        method: ParameterInference = ParameterInference.sum,
) -> ParameterValues:
    """
    Make best efforts to define a Bayesian network model given only the
    evidence from the supplied cross-tables.

    This function infers CPTs for `rvs` using the given `cross_tables`.

    For any two cross-tables `x` and `y` in `cross_tables`, with common random
    variables `rvs` then this function assumes `x.project(rvs) == y.project(rvs)`.
    If this condition does not hold, then best efforts will still be made to
    define a Bayesian network model, however, the resulting parameter values
    may be suboptimal.

    Args:
        rvs: the random variables to define a network structure over.
        cross_tables: available cross-tables to build a model from.
        method: what parameter inference method to use.

    Returns:
        ParameterValues object as a list of CPTs, each CPT can be used to create
        a new factor in the given PGM to make a Bayesian network.
    """
    # Stabilise the given crosstables
    cross_tables: Tuple[CrossTable, ...] = tuple(cross_tables)

    # Heuristically determine an ordering over the random variables
    # which will be used to form the BN structure.
    rv_order: Dict[RandomVariable, int] = _get_rv_order(cross_tables)

    # Make an empty model factor for each random variable.
    model_factors: List[_ModelFactor] = [_ModelFactor(rv) for rv in rvs]

    # Define a Bayesian network structure.
    # Allocate each crosstab to exactly one random variable, the
    # one with the highest rank (i.e. the child).
    for crosstab in cross_tables:
        if len(crosstab.rvs) == 0:
            continue
        sorted_rvs: List[RandomVariable] = sorted(crosstab.rvs, key=(lambda _rv: rv_order[_rv]), reverse=True)
        child: RandomVariable = sorted_rvs[0]
        parents: List[RandomVariable] = sorted_rvs[1:]
        model_factor: _ModelFactor = model_factors[child.idx]
        model_factor.parent_rvs.update(parents)
        model_factor.cross_tables.append(crosstab)

    # Link child factors.
    # When defining a factor, we need to define the child factors first.
    for model_factor in model_factors:
        for parent_rv in model_factor.parent_rvs:
            model_factors[parent_rv.idx].child_factors.append(model_factor)

    # Make the factors, depth first.
    done: Set[int] = set()
    for model_factor in model_factors:
        _infer_cpt(model_factor, done, method)

    # Return the CPTs that define the structure
    return [model_factor.cpt for model_factor in model_factors]


def _infer_cpt(
        model_factor: _ModelFactor,
        done: Set[int],
        method: ParameterInference,
) -> None:
    """
    Depth-first recursively infer the model factors as CPTs.
    This sets `model_factor.cpt` and `model_factor.parent_rvs` for
    the given `model_factor` and its children.
    """
    # Only visit a model factor once.
    child_rv: RandomVariable = model_factor.child_rv
    if child_rv.idx in done:
        return
    done.add(child_rv.idx)

    # Recursively visit the child factors
    for child_model_factor in model_factor.child_factors:
        _infer_cpt(child_model_factor, done, method)

    # Get all relevant cross-tables to set the parameters
    crosstabs: Sequence[CrossTable] = model_factor.cross_tables
    child_crosstabs: Sequence[CrossTable] = [
        child_model_factor.parent_sums
        for child_model_factor in model_factor.child_factors
    ]

    # Get the parameters
    rvs = [child_rv] + list(model_factor.parent_rvs)
    crosstab: CrossTable = _infer_parameter_values(rvs, crosstabs, child_crosstabs, method)
    cpt, parent_sums = cpt_and_parent_sums_from_crosstab(crosstab)
    model_factor.cpt = cpt
    model_factor.parent_sums = parent_sums


def _infer_parameter_values(
        rvs: Sequence[RandomVariable],
        crosstabs: Sequence[CrossTable],
        child_crosstabs: Sequence[CrossTable],
        method: ParameterInference,
) -> CrossTable:
    """
    Make best efforts to infer a probability distribution over the given random variables,
    with evidence from the given cross-tables.

    Returns:
        a CrossTable representing the inferred probability distribution
        (not normalised to sum to one).

    Assumes:
        `rvs` has no duplicates.
    """

    if method == ParameterInference.all:
        # Forced to use all cross-tables with `coalesce_cross_tables`
        projected_crosstabs: List[CrossTable] = [
            crosstab.project(rvs)
            for crosstab in chain(crosstabs, child_crosstabs)
        ]
        return coalesce_cross_tables(projected_crosstabs, rvs)

    # Project crosstables onto rvs, splitting them into complete and partial coverage of `rvs`.
    # Completely covering cross-tables will be summed into `complete_crosstab` while others
    # will be appended to partial_crosstabs.
    complete_crosstab: Optional[CrossTable] = None
    partial_crosstabs: List[CrossTable] = []
    for available_crosstab in chain(crosstabs, child_crosstabs):
        available_rvs: Set[RandomVariable] = set(available_crosstab.rvs)
        if available_rvs.issuperset(rvs):
            to_add: CrossTable = available_crosstab.project(rvs)
            if method == ParameterInference.first:
                # Take the first available solution.
                return to_add
            if complete_crosstab is None:
                complete_crosstab = to_add
            else:
                complete_crosstab.add_all(to_add.items())
        else:
            partial_crosstabs.append(available_crosstab)

    if complete_crosstab is not None:
        # A direct solution was found.
        # Ignore any partially covering cross-tables.
        return complete_crosstab

    # If there are no cross-tables available, the result is empty
    if len(partial_crosstabs) == 0:
        return CrossTable(rvs)

    # There were no cross-tables that completely cover the given random variables.
    # The following algorithm makes best attempts to coalesce a collection of
    # partially covering cross-tables.

    return coalesce_cross_tables(partial_crosstabs, rvs)


def _crostab_str(crosstab: CrossTable) -> str:
    return '(' + ', '.join(repr(rv.name) for rv in crosstab.rvs) + ')'


def _get_rv_order(cross_tables: Sequence[CrossTable]) -> Dict[RandomVariable, int]:
    """
    Determine an order over the given random variables.
    Returns a map from rv to its rank in the order.
    """
    child_parent_map: MapList[RandomVariable, RandomVariable] = MapList()
    for crosstab in cross_tables:
        rvs = crosstab.rvs
        for i in range(len(rvs)):
            child = rvs[i]
            parents = rvs[i + 1:]
            child_parent_map.extend(child, parents)
    order: Dict[RandomVariable, int] = {}
    seen: Set[RandomVariable] = set()
    for child in child_parent_map.keys():
        _get_rv_order_r(child, child_parent_map, seen, order)
    return order


def _get_rv_order_r(
        child: RandomVariable,
        child_parent_map: MapList[RandomVariable, RandomVariable],
        seen: Set[RandomVariable],
        order: Dict[RandomVariable, int],
):
    if child not in seen:
        seen.add(child)
        parents = child_parent_map[child]
        for parent in parents:
            _get_rv_order_r(parent, child_parent_map, seen, order)
        order[child] = len(order)


@dataclass
class _ModelFactor:
    """
    A collection of model factors defines a PGM structure,
    each model factor representing a needed PGM factor.

    Associated with a model factor is a list of crosstabs
    that whose rvs overlap with the rvs of the model factor.
    """

    child_rv: RandomVariable
    parent_rvs: set[RandomVariable] = field(default_factory=set)
    cross_tables: List[CrossTable] = field(default_factory=list)
    child_factors: List[_ModelFactor] = field(default_factory=list)

    # These are set once the parameter values are inferred...
    cpt: Optional[CrossTable] = None
    parent_sums: Optional[CrossTable] = None

    def dump(self, prefix=''):
        """
        For debugging.
        """
        print(f'{prefix}child:', self.child_rv)
        print(f'{prefix}parents:', '{}' if not self.parent_rvs else self.parent_rvs)
        for cross_table in self.cross_tables:
            print(f'{prefix}cross-table:', *cross_table.rvs)
