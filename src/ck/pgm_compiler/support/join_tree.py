from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import List, Set, Callable, Sequence, Tuple

import numpy as np

from ck.pgm import PGM, Factor
from ck.pgm_compiler.support.clusters import Clusters, min_degree, min_fill, \
    min_degree_then_fill, min_fill_then_degree, min_weighted_degree, min_weighted_fill, min_traditional_weighted_fill, \
    ClusterAlgorithm
from ck.utils.np_extras import NDArrayFloat64


@dataclass
class JoinTree:
    """
    This is a recursive data structure representing a join-tree.
    Each node in the join-tree is represented by a JoinTree object.
    """

    # The PGM that this join tree is for.
    pgm: PGM

    # Indexes of random variables in this join tree node
    cluster: Set[int]

    # Child nodes in the join tree
    children: List[JoinTree]

    # Factors of the PGM allocated to this join tree node.
    factors: List[Factor]

    # Indexes of random variables that in both this cluster and the parent's cluster.
    # (Empty if this is the root of the spanning tree).
    separator: Set[int]

    def max_cluster_size(self) -> int:
        """
        Returns:
            the maximum `len(self.cluster)` over self and all children, recursively.
        """
        return max(chain((len(self.cluster),), (child.max_cluster_size() for child in self.children)))

    def max_cluster_weighted_size(self, rv_log_sizes: Sequence[float]) -> float:
        """
        Calculate the maximum cluster weighted size for this cluster and its children.

        Args:
            rv_log_sizes: is an array of random variable sizes, such that
                for a random variable `rv`, `rv_log_sizes[rv.idx] = log2(len(rv))`.

        Returns:
            the maximum `log2` over self and all children, recursively.
        """
        self_weighted_size: float = sum(rv_log_sizes[rv_idx] for rv_idx in self.cluster)
        return max(
            chain(
                (self_weighted_size,),
                (child.max_cluster_weighted_size(rv_log_sizes) for child in self.children)
            )
        )

    def dump(self, *, prefix: str = '', indent: str = '    ', show_factors: bool = True) -> None:
        """
        Print a dump of the Join Tree.
        This is intended for debugging and demonstration purposes.

        Each cluster is printed as: {separator rvs} | {non-separator rvs}.

        Args:
            prefix: optional prefix for indenting all lines.
            indent: additional prefix to use for extra indentation.
            show_factors: if true, the factors of each cluster are shown.
        """
        sep_str = ' '.join(repr(str(self.pgm.rvs[i])) for i in sorted(self.separator))
        rest_str = ' '.join(repr(str(self.pgm.rvs[i])) for i in sorted(self.cluster) if i not in self.separator)
        if len(sep_str) > 0:
            sep_str += ' '
        print(f'{prefix}{sep_str}| {rest_str} (factors: {len(self.factors)})')
        if show_factors:
            for factor in self.factors:
                print(f'{prefix}factor{factor}')
        next_prefix = prefix + indent
        for child in self.children:
            child.dump(prefix=next_prefix, indent=indent, show_factors=show_factors)


# Type for a join tree algorithm: PGM -> JoinTree.
JoinTreeAlgorithm = Callable[[PGM], JoinTree]


def _join_tree_algorithm(pgm_to_clusters: ClusterAlgorithm) -> JoinTreeAlgorithm:
    """
    Helper function for creating a standard JoinTreeAlgorithm
    from a ClusterAlgorithm.

    Args:
        pgm_to_clusters: The clusters method to use.

    Returns:
        a JoinTreeAlgorithm.
    """

    def __join_tree_algorithm(pgm: PGM) -> JoinTree:
        clusters: Clusters = pgm_to_clusters(pgm)
        return clusters_to_join_tree(clusters)

    return __join_tree_algorithm


# standard JoinTreeAlgorithms

MIN_DEGREE: JoinTreeAlgorithm = _join_tree_algorithm(min_degree)
MIN_FILL: JoinTreeAlgorithm = _join_tree_algorithm(min_fill)
MIN_DEGREE_THEN_FILL: JoinTreeAlgorithm = _join_tree_algorithm(min_degree_then_fill)
MIN_FILL_THEN_DEGREE: JoinTreeAlgorithm = _join_tree_algorithm(min_fill_then_degree)
MIN_WEIGHTED_DEGREE: JoinTreeAlgorithm = _join_tree_algorithm(min_weighted_degree)
MIN_WEIGHTED_FILL: JoinTreeAlgorithm = _join_tree_algorithm(min_weighted_fill)
MIN_TRADITIONAL_WEIGHTED_FILL: JoinTreeAlgorithm = _join_tree_algorithm(min_traditional_weighted_fill)


def clusters_to_join_tree(clusters: Clusters) -> JoinTree:
    """
    Construct a join tree from the given random variable clusters.

    A join tree is formed by finding a minimum spanning tree over the clusters
    where the cost between a pair of clusters is the number of random variables
    in common (using separator state space size to break ties).

    Args:
        clusters: the clusters that resulted from graph clusters of a PGM.

    Returns:
        a JoinTree.
    """
    pgm: PGM = clusters.pgm
    cluster_sets: List[Set[int]] = clusters.clusters
    number_of_clusters = len(cluster_sets)

    # Dealing with these cases directly simplifies
    # the spanning tree algorithm implementation.
    if number_of_clusters == 0:
        return JoinTree(pgm, set(), [], [], set())
    elif number_of_clusters == 1:
        return JoinTree(pgm, cluster_sets[0], [], list(pgm.factors), set())

    # Calculate inter-cluster costs for determining the minimum spanning tree
    cost: NDArrayFloat64 = np.zeros((number_of_clusters, number_of_clusters), dtype=np.float64)
    # We will use separator state space size to break ties.
    max_raw_break_cost = sum(pgm.rv_log_sizes) * 1.1  # sum of break costs must be < 1
    break_cost = [cost / max_raw_break_cost for cost in pgm.rv_log_sizes]
    for i in range(number_of_clusters):
        cluster_i = cluster_sets[i]
        for j in range(i + 1, number_of_clusters):
            cluster_j = cluster_sets[j]
            separator = cluster_i.intersection(cluster_j)
            cost[i, j] = cost[j, i] = -len(separator) + sum(break_cost[rv_idx] for rv_idx in separator)

    # Make the spanning tree over the clusters
    root_custer_index: int
    children: List[List[int]]
    children, root_custer_index = _make_spanning_tree_small_root(cost, clusters.clusters)

    # Allocate each PGM factor to a cluster
    cluster_factors: List[List[Factor]] = [[] for _ in range(number_of_clusters)]
    ordered_indexed_clusters = list(enumerate(cluster_sets))
    ordered_indexed_clusters.sort(key=lambda idx_c: len(idx_c[1]))  # sort from smallest to largest cluster
    for factor in pgm.factors:
        rv_indexes = frozenset(rv.idx for rv in factor.rvs)
        for cluster_index, cluster in ordered_indexed_clusters:
            if rv_indexes.issubset(cluster):
                cluster_factors[cluster_index].append(factor)
                break

    return _form_join_tree_r(pgm, root_custer_index, set(), children, cluster_sets, cluster_factors)


_INF = float('inf')


def _make_spanning_tree_small_root(cost: NDArrayFloat64, clusters: List[Set[int]]) -> Tuple[List[List[int]], int]:
    """
    Construct a minimum spanning tree over the clusters, where the root is the cluster with
    the smallest number of random variable.

    Args:
        cost: is an N x N matrix of costs between N clusters.
        clusters: is a list of N clusters, each cluster is a set of random variable indices.

    Returns:
        (spanning_tree, root_index)

        spanning_tree: is a spanning tree represented as a list of nodes, the list is coindexed with
        the given cost matrix, each node is a list of children, each child being
        represented as an index into the list of nodes.

        root_index: is the index the chosen root of the spanning tree.
    """
    root_custer_index: int = 0
    root_size: int = len(clusters[root_custer_index])
    for i, cluster in enumerate(clusters[1:], start=1):
        if len(clusters[root_custer_index]) < root_size:
            root_custer_index = i
            root_size: int = len(cluster)

    children: List[List[int]] = _make_spanning_tree_at_root(cost, root_custer_index)
    return children, root_custer_index


def _make_spanning_tree_arbitrary_root(cost: NDArrayFloat64) -> Tuple[List[List[int]], int]:
    """
    Construct a minimum spanning tree over the clusters, starting at an arbitrary root.

    Args:
        cost: is an N x N matrix of costs between N clusters.

    Returns:
        (spanning_tree, root_index)

        spanning_tree: is a spanning tree represented as a list of nodes, the list is coindexed with
        the given cost matrix, each node is a list of children, each child being
        represented as an index into the list of nodes.

        root_index: is the index the chosen root of the spanning tree.
    """
    root_index: int = 0
    spanning_tree: List[List[int]] = _make_spanning_tree_at_root(cost, root_index)
    return spanning_tree, root_index


def _make_spanning_tree_at_root(
        cost: NDArrayFloat64,
        root_custer_index: int,
) -> List[List[int]]:
    """
    Construct a minimum spanning tree over the clusters, starting at the given root.

    Args:
        cost: and nxn matrix where n is the number of clusters and cost[i, j]
            gives the cost between clusters i and j.
        root_custer_index: a nominated root cluster to be the root of the tree.

    Returns:
        a spanning tree represented as a list of nodes, the list is coindexed with
        the given cost matrix, each node is a list of children, each child being
        represented as an index into the list of nodes. The root node is the
        index `root_custer_index` as passed to this function.
    """
    number_of_clusters: int = cost.shape[0]

    # clusters left to process.
    remaining: List[int] = list(range(number_of_clusters))

    # clusters that have been processed.
    included: List[int] = []

    def remove_remaining(_remaining_index: int) -> None:
        # Remove the `remaining` element at the given index location.
        remaining[_remaining_index] = remaining[-1]
        remaining.pop()

    # Move root from `remaining` to `included`
    included.append(root_custer_index)
    remove_remaining(root_custer_index)  # assumes remaining[root_custer_index] = root_custer_index

    # Data structure to collect the results.
    children: List[List[int]] = [[] for _ in range(number_of_clusters)]

    while True:
        min_i: int = 0
        min_j: int = 0
        min_j_pos: int = 0
        min_c: float = _INF
        for i in included:
            for j_pos, j in enumerate(remaining):
                c: float = cost.item(i, j)
                if c < min_c:
                    min_c = c
                    min_i = i
                    min_j = j
                    min_j_pos = j_pos

        # Record the child and move remaining_idx from 'remaining' to 'included'.
        children[min_i].append(min_j)
        if len(remaining) == 1:
            # That was the last one.
            return children

        # Update `remaining` and `included`
        remove_remaining(min_j_pos)
        included.append(min_j)


def _form_join_tree_r(
        pgm: PGM,
        cluster_index: int,
        parent_cluster: Set[int],
        children: Sequence[List[int]],
        clusters: Sequence[Set[int]],
        cluster_factors: List[List[Factor]],
) -> JoinTree:
    """
    Recursively build a JoinTree from the spanning tree `children`.
    This function merely pull the corresponding component from the
    arguments to make a JoinTree object, doing this recursively
    for the children.

    Args:
        pgm: the source PGM for the join tree.
        cluster_index: index for the node we are processing (current root). This
            indexes into `children`, `clusters`, and `cluster_factors`.
        parent_cluster: set of random variable indices in the parent cluster.
        children: list of spanning tree nodes, as per `_make_spanning_tree_at_root` result.
        clusters: list of clusters, each cluster is a set of random variable indices.
        cluster_factors: assignment of factors to clusters.
    """
    cluster: Set[int] = clusters[cluster_index]
    factors: List[Factor] = cluster_factors[cluster_index]
    children = [
        _form_join_tree_r(pgm, child, cluster, children, clusters, cluster_factors)
        for child in children[cluster_index]
    ]
    separator: Set[int] = parent_cluster.intersection(cluster)
    return JoinTree(
        pgm,
        cluster,
        children,
        factors,
        separator,
    )
