from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import islice
from typing import Iterator, Optional, FrozenSet

from ck.circuit import CircuitNode
from ck.pgm_circuit import PGMCircuit
from ck.pgm_compiler.support.circuit_table import CircuitTable, product, sum_out
from ck.pgm_compiler.support.factor_tables import make_factor_tables, FactorTables
from ck.pgm_compiler.support.join_tree import *

_NEG_INF = float('-inf')

DEFAULT_PRODUCT_SEARCH_LIMIT: int = 1000


def compile_pgm(
        pgm: PGM,
        const_parameters: bool = True,
        *,
        algorithm: JoinTreeAlgorithm = MIN_FILL_THEN_DEGREE,
        limit_product_tree_search: int = DEFAULT_PRODUCT_SEARCH_LIMIT,
        pre_prune_factor_tables: bool = False,
) -> PGMCircuit:
    """
    Compile the PGM to an arithmetic circuit, using factor elimination.

    When forming the product of factors within join tree nodes,
    this method searches all practical binary trees for forming products,
    up to the given limit, `limit_product_tree_search`. The minimum is 1.

    Conforms to the `PGMCompiler` protocol.

    Args:
        pgm: The PGM to compile.
        const_parameters: If true, the potential function parameters will be circuit
            constants, otherwise they will be circuit variables.
        algorithm: algorithm to get a join tree.
        limit_product_tree_search: limit on number of product trees to consider.
        pre_prune_factor_tables: if true, then heuristics will be used to remove any provably zero row.

    Returns:
        a PGMCircuit object.

    Raises:
        ValueError if `limit_product_tree_search` is not > 0.
    """
    join_tree: JoinTree = algorithm(pgm)
    return join_tree_to_circuit(
        join_tree,
        const_parameters,
        limit_product_tree_search,
        pre_prune_factor_tables,
    )


def compile_pgm_best_jointree(
        pgm: PGM,
        const_parameters: bool = True,
        *,
        limit_product_tree_search: int = DEFAULT_PRODUCT_SEARCH_LIMIT,
        pre_prune_factor_tables: bool = False,
) -> PGMCircuit:
    """
    Try multiple elimination heuristics, and use the join tree that has
    the smallest maximum cluster size.

    Conforms to the `PGMCompiler` protocol.

    Args:
        pgm: The PGM to compile.
        const_parameters: If true, the potential function parameters will be circuit
            constants, otherwise they will be circuit variables.
        limit_product_tree_search: limit on number of product trees to consider.
        pre_prune_factor_tables: if true, then heuristics will be used to remove any provably zero row.

    Returns:
        a PGMCircuit object.

    Raises:
        ValueError if `limit_product_tree_search` is not > 0.
    """
    # Get the smallest cluster sequence for a list of possibles.
    algorithms: Sequence[ClusterAlgorithm] = [
        min_degree,
        min_fill,
        min_degree_then_fill,
        min_fill_then_degree,
        min_weighted_degree,
        min_weighted_fill,
        min_traditional_weighted_fill,
    ]
    rv_log_sizes: Sequence[float] = pgm.rv_log_sizes
    best_clusters: Clusters = algorithms[0](pgm)
    best_size = best_clusters.max_cluster_weighted_size(rv_log_sizes)
    for algorithm in algorithms[1:]:
        clusters: Clusters = algorithm(pgm)
        size = clusters.max_cluster_weighted_size(rv_log_sizes)
        if size < best_size:
            best_size = size
            best_clusters = clusters

    join_tree: JoinTree = clusters_to_join_tree(best_clusters)
    return join_tree_to_circuit(
        join_tree,
        const_parameters,
        limit_product_tree_search,
        pre_prune_factor_tables,
    )


def join_tree_to_circuit(
        join_tree: JoinTree,
        const_parameters: bool = True,
        limit_product_tree_search: int = DEFAULT_PRODUCT_SEARCH_LIMIT,
        pre_prune_factor_tables: bool = False,
) -> PGMCircuit:
    """
    Construct a PGMCircuit from a join-tree.

    Args:
        join_tree: a join tree for a PGM.
        const_parameters: If true, the potential function parameters will be circuit
            constants, otherwise they will be circuit variables.
        limit_product_tree_search: limit on number of product trees to consider.
        pre_prune_factor_tables: if true, then heuristics will be used to remove any provably zero row.

    Returns:
        an arithmetic circuit and slot map, as a PGMCircuit object.

    Raises:
        ValueError if `limit_product_tree_search` is not > 0.
    """
    if limit_product_tree_search <= 0:
        raise ValueError('limit_product_tree_search must be > 0')

    pgm: PGM = join_tree.pgm
    factor_tables: FactorTables = make_factor_tables(
        pgm=pgm,
        const_parameters=const_parameters,
        multiply_indicators=True,
        pre_prune_factor_tables=pre_prune_factor_tables,
    )

    top_table: CircuitTable = _circuit_tables_from_join_tree(
        factor_tables,
        join_tree,
        limit_product_tree_search,
    )
    top: CircuitNode = top_table.top()
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


def _circuit_tables_from_join_tree(
        factor_tables: FactorTables,
        join_tree: JoinTree,
        limit_product_tree_search: int,
) -> CircuitTable:
    """
    This is a basic algorithm for constructing a circuit table from a join tree.
    Algorithm synopsis:
    1) Get a CircuitTable for each factor allocated to this join tree node, and
       for each child of the join tree node (recursive call to _circuit_tables_from_join_tree).
    2) Form a binary tree of the collected circuit tables.
    3) Perform table products and sum-outs for each node in the binary tree, which should
       leave a single circuit table with a single row.
    """
    # Get all the factors to combine.
    factors: List[CircuitTable] = list(
        chain(
            (
                # The PGM factors allocated to this join tree node
                factor_tables.get_table(factor)
                for factor in join_tree.factors
            ),
            (
                # The children of this join tree node
                _circuit_tables_from_join_tree(factor_tables, child, limit_product_tree_search)
                for child in join_tree.children
            ),
        )
    )

    # The usual join tree approach just forms the product all the tables in `factors`.
    # The tree width is not affected by the order of products, however some orders
    # lead to smaller numbers of arithmetic operations.
    #
    # If `limit_product_tree_search > 1`, then heuristics are used
    # reduce the number of arithmetic operations.

    # Deal with the special case: zero factors
    if len(factors) == 0:
        circuit = factor_tables.circuit
        if len(join_tree.separator) == 0:
            # table one
            return CircuitTable(circuit, (), (((), circuit.one),))
        else:
            # table zero
            return CircuitTable(circuit, tuple(join_tree.separator), ())

    # Analise different ways to combine the factors
    # This method potentially examines all possible trees, O(len(factors)!),
    # which may need to be improved!
    # Trees that result in rvs to be summed out early are scored more highly.

    rv_log_sizes: Sequence[float] = join_tree.pgm.rv_log_sizes
    best_score = _NEG_INF
    best_tree = None
    for tree in islice(_iterate_trees(factors, join_tree.separator), limit_product_tree_search):
        score = tree.score(rv_log_sizes)
        if score > best_score:
            best_score = score
            best_tree = tree

    # The tree knows how to form products and perform sum-outs.
    return best_tree.get_table()


class _Product(ABC):
    """
    A node in a binary product tree.

    A node is either a _ProductLeaf, holding a single CircuitTable,
    or is a _ProductInterior, which has exactly two children.
    """

    def __init__(self, available: Set[int]):
        """
        Construct a node in a binary product tree.

        Args:
            available: the rvs that are available (prior to sum-out)
            after the product is formed.
        """
        self.available: Set[int] = available
        self.sum_out: Set[int] = set()

    @abstractmethod
    def set_sum_out(self, need: Set[int]) -> None:
        """
        Set the self.sum_out, based on what rvs are needed.

        Args:
            need: what rvs are require to be supplied by this node
                after the product is formed. This will be a subset
                of `self.available`.
        """
        ...

    @abstractmethod
    def score(self, rv_log_sizes: Sequence[float]) -> float:
        """
        Heuristically score a tree (assuming set_sum_out has been called).
        """
        ...

    @abstractmethod
    def get_table(self) -> CircuitTable:
        """
        Returns:
            The circuit table (after products and sum-outs) implied
            by this node.
        """
        ...


@dataclass
class _ProductLeaf(_Product):

    def __init__(self, table: CircuitTable):
        super().__init__(set(table.rv_idxs))
        self.table: CircuitTable = table

    def set_sum_out(self, need: Set[int]) -> None:
        self.sum_out = self.available.difference(need)

    def score(self, rv_log_sizes: Sequence[float]) -> float:
        return sum(rv_log_sizes[i] for i in self.sum_out)

    def get_table(self) -> CircuitTable:
        return sum_out(self.table, self.sum_out)


@dataclass
class _ProductInterior(_Product):

    def __init__(self, x: _Product, y: _Product):
        super().__init__(x.available.union(y.available))
        self.x: _Product = x
        self.y: _Product = y

    def set_sum_out(self, need: Set[int]) -> None:
        x = self.x
        y = self.y
        x_y_common: Set[int] = x.available.intersection(y.available)
        x_need: Set[int] = x.available.intersection(chain(need, x_y_common))
        y_need: Set[int] = y.available.intersection(chain(need, x_y_common))
        self.x.set_sum_out(x_need)
        self.y.set_sum_out(y_need)
        self.sum_out = x_need.union(y_need).difference(need)

    def score(self, rv_log_sizes: Sequence[float]) -> float:
        x_score = self.x.score(rv_log_sizes)
        y_score = self.y.score(rv_log_sizes)
        return sum(rv_log_sizes[i] for i in self.sum_out) + (x_score + y_score) * 2

    def get_table(self) -> CircuitTable:
        return sum_out(product(self.x.get_table(), self.y.get_table()), self.sum_out)


def _iterate_trees(factors: List[CircuitTable], separator: Set[int]) -> Iterator[_Product]:
    """
    Iterate over all possible binary trees that form the product of the given factors.

    Args:
        factors: The list of factors to be in the product.
        separator: What rvs the resulting product needs to be projected onto.

    Returns:
        An iterator over binary product trees.

    Assumes:
        There is at least one factor.
    """
    leaves = [_ProductLeaf(table) for table in factors]
    for tree in _iterate_trees_r(leaves):
        tree.set_sum_out(separator)
        yield tree


def _iterate_trees_r(factors: List[_Product]) -> Iterator[_Product]:
    """
    Recursive support function for _iterate_trees.

    This will form the products, but not will not set the
    `sum_out` field as that can only be done once a tree is fully formed.

    Args:
        factors: The list of factors to be in the product.

    Returns:
        An iterator over binary product trees.

    Assumes:
        There is at least one factor.
    """

    # Use heuristics to reduce the number of arithmetic operations.
    # If the rvs of one factor is a subset of another factor, form their
    # product, preferring to product factors with small numbers of rvs.

    # Sort factors by number or rvs (in increasing order).
    sorted_factors: List[Tuple[FrozenSet[int], Optional[_Product]]] = sorted(
        (
            (frozenset(factor.available), factor)
            for factor in factors
        ),
        key=lambda _x: _x[0]
    )

    # Product any factor who's rvs are a subset of another factor.
    i: int
    j: int
    for i, (rvs_idxs, factor) in enumerate(sorted_factors):
        for j, (other_rvs_idxs, other_factor) in enumerate(sorted_factors[i + 1:], start=i + 1):
            if other_rvs_idxs.issuperset(rvs_idxs):
                sorted_factors[j] = (other_rvs_idxs, _ProductInterior(other_factor, factor))
                sorted_factors[i] = (rvs_idxs, None)
                break
    factors = [factor for _, factor in sorted_factors if factor is not None]

    if len(factors) == 1:
        yield factors[0]
    elif len(factors) == 2:
        yield _ProductInterior(*factors)
    else:
        for i in range(len(factors)):
            for j in range(i):
                copy: List[_Product] = factors.copy()
                x = copy.pop(i)
                y = copy.pop(j)
                copy.append(_ProductInterior(x, y))
                for tree in _iterate_trees_r(copy):
                    yield tree
