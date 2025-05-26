from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Dict, Optional, List, Sequence, Tuple, Set

from ck.circuit import Circuit, CircuitNode
from ck.pgm import PGM
from ck.pgm_circuit import PGMCircuit
from ck.pgm_compiler.support import clusters
from ck.pgm_compiler.support.circuit_table import CircuitTable
from ck.pgm_compiler.support.clusters import ClusterAlgorithm
from ck.pgm_compiler.support.factor_tables import make_factor_tables, FactorTables
from ck.utils.iter_extras import combos

# Standard cluster algorithms.
MIN_DEGREE: ClusterAlgorithm = clusters.min_degree
MIN_FILL: ClusterAlgorithm = clusters.min_fill
MIN_DEGREE_THEN_FILL: ClusterAlgorithm = clusters.min_degree_then_fill
MIN_FILL_THEN_DEGREE: ClusterAlgorithm = clusters.min_fill_then_degree
MIN_WEIGHTED_DEGREE: ClusterAlgorithm = clusters.min_weighted_degree
MIN_WEIGHTED_FILL: ClusterAlgorithm = clusters.min_weighted_fill
MIN_TRADITIONAL_WEIGHTED_FILL: ClusterAlgorithm = clusters.min_traditional_weighted_fill


def compile_pgm(
        pgm: PGM,
        const_parameters: bool = True,
        *,
        algorithm: ClusterAlgorithm = MIN_FILL_THEN_DEGREE,
        pre_prune_factor_tables: bool = False,
) -> PGMCircuit:
    """
    Compile the PGM to an arithmetic circuit, using recursive conditioning.

    Conforms to the `PGMCompiler` protocol.

    Args:
        pgm: The PGM to compile.
        const_parameters: If true, the potential function parameters will be circuit
            constants, otherwise they will be circuit variables.
        algorithm: algorithm to get an elimination order.
        pre_prune_factor_tables: if true, then heuristics will be used to remove any provably zero row.

    Returns:
        a PGMCircuit object.
    """
    elimination_order: Sequence[int] = algorithm(pgm).eliminated
    factor_tables: FactorTables = make_factor_tables(
        pgm=pgm,
        const_parameters=const_parameters,
        multiply_indicators=True,
        pre_prune_factor_tables=pre_prune_factor_tables,
    )

    if pgm.number_of_factors == 0:
        # Deal with special case: no factors
        top: CircuitNode = factor_tables.circuit.const(1)
    else:
        dtree: _DTree = _make_dtree(elimination_order, factor_tables)
        states: List[Sequence[int]] = [tuple(range(len(rv))) for rv in pgm.rvs]
        top: CircuitNode = dtree.make_circuit(states, factor_tables.circuit)

    top.circuit.remove_unreachable_op_nodes(top)

    return PGMCircuit(
        rvs=tuple(pgm.rvs),
        conditions=(),
        circuit_top=top,
        number_of_indicators=factor_tables.number_of_indicators,
        number_of_parameters=factor_tables.number_of_parameters,
        slot_map=factor_tables.slot_map,
        parameter_values=factor_tables.parameter_values,
    )


def _make_dtree(elimination_order: Sequence[int], factor_tables: FactorTables) -> _DTree:
    if len(factor_tables.tables) == 0:
        return _DTreeLeaf(CircuitTable(factor_tables.circuit, (), ()))

    # Populate `trees` with all the leaves
    trees: List[_DTree] = [_DTreeLeaf(table) for table in factor_tables.tables]

    # join trees by elimination random variable
    for rv_index in elimination_order:
        next_trees: List[_DTree] = []
        to_join: List[_DTree] = []
        for tree in trees:
            if rv_index in tree.vars:
                to_join.append(tree)
            else:
                next_trees.append(tree)
        if len(to_join) >= 2:
            while len(to_join) > 1:
                # join the two shallowest trees
                to_join.sort(key=lambda t: -t.depth())
                x = to_join.pop()
                y = to_join.pop()
                to_join.append(_DTreeInterior(x, y))
            next_trees.append(to_join[0])
            trees = next_trees

    # Make sure there is only one tree
    while len(trees) > 1:
        x = trees.pop(0)
        y = trees.pop(0)
        trees.append(_DTreeInterior(x, y))

    root = trees[0]
    root.update_cutset()
    return root


class _DTree(ABC):
    """
    A node in a binary decomposition tree.

    A node is either a _DTreeLeaf, holding a single CircuitTable,
    or is a _DTreeInterior, which has exactly two children.
    """

    def __init__(self, vars_idxs: Set[int]):
        self.vars: Set[int] = vars_idxs
        self.cutset: Sequence[int] = ()
        self.context: Sequence[int] = ()

    @abstractmethod
    def update_cutset(self, acutset: Iterable[int] = ()) -> None:
        """
        After the d-tree is defined, call `update_cutset` on the root
        to ensure all fields are properly set.
        """
        ...

    @abstractmethod
    def make_circuit(self, states: List[Sequence[int]], circuit: Circuit) -> CircuitNode:
        """
        After the d-tree is defined and cutsets are updated,
        construct a circuit using recursive conditioning.
        """
        ...

    @abstractmethod
    def depth(self) -> int:
        """
        Tree depth.
        """


@dataclass
class _DTreeLeaf(_DTree):

    def __init__(self, table: CircuitTable):
        super().__init__(set(table.rv_idxs))
        self.table: CircuitTable = table

    def update_cutset(self, acutset: Iterable[int] = ()) -> None:
        pass

    def make_circuit(self, states: List[Sequence[int]], circuit: Circuit) -> CircuitNode:
        table = self.table

        key_states: List[Sequence[int]] = [
            states[rv_idx]
            for rv_idx in table.rv_idxs
        ]
        to_sum: List[CircuitNode] = list(
            filter(
                (lambda n: n is not None),
                (table.get(key) for key in combos(key_states))
            )
        )
        return circuit.optimised_add(to_sum)

    def depth(self) -> int:
        return 1


@dataclass
class _DTreeInterior(_DTree):

    def __init__(self, x: _DTree, y: _DTree):
        super().__init__(x.vars.union(y.vars))
        self.x: _DTree = x
        self.y: _DTree = y
        self.cache: Dict[Tuple[int, ...], CircuitNode] = {}

    def update_cutset(self, acutset: Iterable[int] = ()) -> None:
        cutset: Set[int] = self.x.vars.intersection(self.y.vars).difference(acutset)
        self.cutset = tuple(cutset)
        self.context = tuple(self.vars.intersection(acutset))

        next_acutset = cutset.union(acutset)
        self.x.update_cutset(next_acutset)
        self.y.update_cutset(next_acutset)

    def make_circuit(self, states: List[Sequence[int]], circuit: Circuit) -> CircuitNode:

        assert all(len(states[rv_idx]) == 1 for rv_idx in self.context), 'consistency check'
        context_key: Tuple[int, ...] = tuple(
            states[rv_idx][0]
            for rv_idx in self.context
        )

        cache: Optional[CircuitNode] = self.cache.get(context_key)
        if cache is not None:
            return cache

        cutset = self.cutset
        key_states: List[Sequence[int]] = [
            states[rv_idx]
            for rv_idx in cutset
        ]
        to_sum: List[CircuitNode] = []
        for key in combos(key_states):
            # Update the evidence with the keys
            next_states = states.copy()
            for s, i in zip(key, cutset):
                next_states[i] = (s,)

            # Recursively call
            x_node = self.x.make_circuit(next_states, circuit)
            y_node = self.y.make_circuit(next_states, circuit)
            to_sum.append(circuit.optimised_mul((x_node, y_node)))

        result = circuit.optimised_add(to_sum)
        self.cache[context_key] = result
        return result

    def depth(self) -> int:
        return max(self.x.depth(), self.y.depth())
