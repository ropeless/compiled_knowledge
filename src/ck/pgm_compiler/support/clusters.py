"""
Graph analysis to identify clusters using elimination heuristics.
"""
from __future__ import annotations

from typing import Set, Iterable, Callable, Iterator, Tuple, List, overload, Sequence

from ck.pgm import PGM

# A VEObjective is a variable elimination objective function.
# An objective function is a function from a random variable index (int)
# to an objective value (float or int). This is used to select
# a random variable to eliminate in `ve_greedy_min`.
VEObjective = Callable[[int], int | float]


def ve_fixed(clusters: Clusters, order: Iterable[int]) -> None:
    """
    Apply the given fixed elimination order to the elimination tree.

    Args:
        clusters: a clusters object with uneliminated random variables.
        order: the order of variable elimination.

    Assumes:
        * All rv indexes in `order` are also in `clusters.uneliminated`.
        * There are no duplicates in `order``.
    """
    for rv_index in order:
        clusters.eliminate(rv_index)


def ve_greedy_min(
        clusters: Clusters,
        objective: VEObjective | Tuple[VEObjective, ...],
        use_twig_prefix: bool = True,
        use_optimal_prefix: bool = False,
) -> None:
    """
    The greedy variable elimination heuristic.

    The objective is a function from (eliminable: Clusters, var_idx: int) to
    which should return an objective value (to greedily minimise by the method).
    The objective may be a tuple of objective functions for tie breaking.

    Args:
        clusters: a clusters object with uneliminated random variables.
        objective: the objective function ( or a tuple of objective functions) to minimise each iteration.
        use_twig_prefix: if true, then `twig_prefix` is used to eliminate any
            candidate random variable prior selecting random variables using the objective function.
        use_optimal_prefix: if true, then `optimal_prefix` is used to eliminate any
            candidate random variable prior selecting random variables using the objective function.
    """
    uneliminated: Set[int] = clusters.uneliminated

    if isinstance(objective, tuple):
        def __objective(_rv_index: int) -> Tuple[float | int, ...]:
            return tuple(f(_rv_index) for f in objective)
    else:
        __objective = objective

    while len(uneliminated) > 1:

        if use_twig_prefix:
            twig_prefix(clusters)
            if len(uneliminated) <= 1:
                break

        if use_optimal_prefix:
            optimal_prefix(clusters)
            if len(uneliminated) <= 1:
                break

        min_iter: Iterator[int] = iter(uneliminated)
        min_rv_index = next(min_iter)
        min_obj = __objective(min_rv_index)
        for rv_index in min_iter:
            obj = __objective(rv_index)
            if obj < min_obj:
                min_rv_index = rv_index
                min_obj = obj
        clusters.eliminate(min_rv_index)

    if len(uneliminated) > 0:
        # eliminate the last rv
        clusters.eliminate(next(iter(uneliminated)))


def twig_prefix(clusters: Clusters) -> None:
    """
    Eliminate all random variables with degree zero or one.
    """

    def get_rvs(degree: int) -> List[int]:
        return [
            _rv_index
            for _rv_index in clusters.uneliminated
            if clusters.degree(_rv_index) == degree
        ]

    for rv_index in get_rvs(degree=0):
        clusters.eliminate(rv_index)

    while len(clusters.uneliminated) > 0:
        eliminating = get_rvs(degree=1)
        if len(eliminating) == 0:
            break
        for rv_index in eliminating:
            clusters.eliminate(rv_index)


def optimal_prefix(clusters: Clusters) -> None:
    """
    Eliminate all random variables that are guaranteed to be optimal (in resulting tree width).

    See Adnan Darwiche, 2009, Modeling and Reasoning with Bayesian Networks, p207.
    """

    def _get_lower_bound() -> int:
        # Return a lower bound on the tree width for the current clusters.
        return max(
            max(
                (len(clusters.connections(_rv_index)) for _rv_index in clusters.uneliminated),
                default=0
            ) - 1,
            0
        )

    prev_number_uneliminated: int = len(clusters.uneliminated) + 1

    while prev_number_uneliminated > len(clusters.uneliminated):
        prev_number_uneliminated = len(clusters.uneliminated)
        low: int = _get_lower_bound()
        to_eliminate: Set[int] = set()
        for rv_index in clusters.uneliminated:
            fill: int = clusters.fill(rv_index)
            if fill == 0:
                # simplical rule: no fill edges
                to_eliminate.add(rv_index)
            elif fill == 1 and clusters.degree(rv_index) <= low:
                # almost simplical rule: one fill edge and degree <= low
                to_eliminate.add(rv_index)

        # Perform eliminations
        for rv_index in to_eliminate:
            clusters.eliminate(rv_index)

        low: int = _get_lower_bound()
        if low >= 3:
            to_eliminate: Set[int] = set()
            for rv_index_i in clusters.uneliminated:
                if clusters.degree(rv_index_i) == 3:
                    i_neighbours: Set[int] = clusters.connections(rv_index_i)

                    # buddy rule: two joined nodes with degree 3 and sam neighbours
                    for rv_index_j in i_neighbours:
                        if clusters.degree(rv_index_j) == 3:
                            j_neighbours: Set[int] = clusters.connections(rv_index_j)
                            if i_neighbours.difference([rv_index_j]) == j_neighbours.difference([rv_index_i]):
                                to_eliminate.add(rv_index_i)
                                to_eliminate.add(rv_index_j)

                    # check cube rule: i, a, b, c form a cube
                    if len(i_neighbours) == 3:
                        if all(clusters.degree(rv_index) == 3 for rv_index in i_neighbours):
                            a, b, c = tuple(i_neighbours)
                            ab = clusters.connections(a).intersection(clusters.connections(a))
                            ac = clusters.connections(a).intersection(clusters.connections(c))
                            bc = clusters.connections(b).intersection(clusters.connections(c))
                            if len(ab) == 1 and len(ac) == 1 and len(bc) == 1:
                                to_eliminate.add(rv_index_i)
                                to_eliminate.add(a)
                                to_eliminate.add(b)
                                to_eliminate.add(c)

            # Perform eliminations
            for rv_index in to_eliminate:
                clusters.eliminate(rv_index)


class Clusters:
    """
    A Clusters object holds the state of a connection graph while
    eliminating variables to construct clusters for a PGM graph.

    The Clusters object can either be "in-progress" where `len(Clusters.uneliminated) > 0`,
    or be "completed" where `len(Clusters.uneliminated) == 0`.

    See Adnan Darwiche, 2009, Modeling and Reasoning with Bayesian Networks, p164.
    """

    def __init__(self, pgm: PGM, maximal_clusters_only: bool = True):
        """
        Args:
            pgm: source PGM defining initial connection graph.
            maximal_clusters_only: if true, then any subsumed cluster will be incorporated
                into its subsuming cluster (once all random variables are eliminated).
        """
        self._pgm: PGM = pgm
        self._uneliminated: Set[int] = {rv.idx for rv in pgm.rvs}
        self._eliminated: List[int] = []
        self._rv_log_sizes: Sequence[float] = pgm.rv_log_sizes
        self._maximal_clusters_only = maximal_clusters_only

        # Create a connection set for each random variable.
        # The connection set keeps track of what _other_ random variable it's connected to (via factors).
        # I.e., the connections define an interaction graph.
        connections: List[Set[int]] = [set() for _ in range(pgm.number_of_rvs)]
        for factor in pgm.factors:
            rv_indexes = [rv.idx for rv in factor.rvs]
            for index in rv_indexes:
                connections[index].update(rv_indexes)
        for index, rv_connections in enumerate(connections):
            rv_connections.discard(index)
        self._connections = connections

        # Deal with the case of an empty PGM.
        if len(self._uneliminated) == 0:
            self._finish_elimination()

    @property
    def pgm(self) -> PGM:
        """
        Returns:
            the PGM that these clusters refer to.
        """
        return self._pgm

    @property
    def eliminated(self) -> List[int]:
        """
        Get the list of eliminated random variables (as random variable
        indices, in elimination order).

        Assumes:
            * The returned list will not be modified by the caller.

        Returns:
            the indexes of eliminated random variables, in elimination order.
        """
        return self._eliminated

    @property
    def uneliminated(self) -> Set[int]:
        """
        Get the set of uneliminated random variables (as random variable indices).

        Assumes:
            * The returned set will not be modified by the caller.

        Returns:
            the set of random variable indexes that are yet to be eliminated.
        """
        return self._uneliminated

    def connections(self, rv_index: int) -> Set[int]:
        """
        Get the current graph connections of a random variable.

        Args:
            rv_index: The index of the random variable being queried.

        Returns:
            the set of random variable indexes that connected to the
            given indexed random variable.

        Assumes:
            * Not all random variables are eliminated.
            * `rv_idx` is in `self.uneliminated()`.
            * The returned set will not be modified by the caller.
        """
        assert len(self._uneliminated) > 0, 'only makes sense while eliminating'
        return self._connections[rv_index]

    @property
    def clusters(self) -> List[Set[int]]:
        """
        Get the clusters that are a result of eliminating all random variables.
        This only makes sense once all random variables are eliminated.

        Assumes:
            * All random variables are eliminated.
            * The returned list and sets will not be modified by the caller.

        Returns:
            list of all clusters, each cluster is a set of random variable indexes.
        """
        assert len(self._uneliminated) == 0, 'only makes sense when completed eliminating'
        return self._connections

    def max_cluster_size(self) -> int:
        """
        Calculate the maximum cluster size over all clusters.

        Returns:
            the maximum `len(cluster)` over all clusters.
        """
        return max((len(cluster) for cluster in self.clusters), default=0)

    def max_cluster_weighted_size(self, rv_log_sizes: Sequence[float]) -> float:
        """
        Calculate the maximum cluster weighted size over all clusters.

        Args:
            rv_log_sizes: is an array of random variable sizes, such that
                for a random variable `rv`, `rv_log_sizes[rv.idx] = log2(len(rv))`,
                e.g., `self.pgm.rv_log_sizes`.
        Returns:
            the maximum `sum(rv_log_sizes[rv_idx] for rv_idx in cluster)` over all clusters.
        """
        return max(
            (
                sum(rv_log_sizes[rv_idx] for rv_idx in cluster)
                for cluster in self.clusters
            ),
            default=0
        )

    def eliminate(self, rv_index: int) -> None:
        """
        Perform one step of variable elimination.

        A cluster will be identified (either existing or new) to cover the eliminated
        random variable and any other interacting random variables according to
        the factors of the3 PGM. The elimination will be recorded in the identified cluster.

        Assumes:
            `rv_idx` is in `self.uneliminated()`.
        """

        # record that the rv is eliminated now
        self._uneliminated.remove(rv_index)  # may raise a KeyError.
        self._eliminated.append(rv_index)

        # Get all rvs connected to the rv being eliminated.
        # For every rv mentioned, connect to all the others.
        # This adds fill edges to connections.
        mentioned_rvs: Set[int] = self._connections[rv_index]
        for mentioned_index in mentioned_rvs:
            rv_connections = self._connections[mentioned_index]
            rv_connections.update(mentioned_rvs)
            rv_connections.discard(mentioned_index)
            rv_connections.discard(rv_index)

        if len(self._uneliminated) == 0:
            self._finish_elimination()

    def degree(self, rv_index: int) -> int:
        """
        What is the degree of the random variable with the given index
        given the current state of eliminations.
        Mathematically equivalent to `len(self.connections(rv_index))`.
        """
        assert len(self._uneliminated) > 0, 'only makes sense while eliminating'
        return len(self._connections[rv_index])

    def fill(self, rv_index: int) -> int:
        """
        What number of  new fill edges are created if eliminating the random variable with
        the given index given the current state of eliminations.
        """
        assert len(self._uneliminated) > 0, 'only makes sense while eliminating'
        return self._fill_count(
            rv_index,
            self._add_one,
            self._identity,
        )

    def weighted_degree(self, rv_index: int) -> float:
        """
        What is the total weight of fill edges are created if eliminating the random variable with
        the given index given the current state of eliminations.
        """
        assert len(self._uneliminated) > 0, 'only makes sense while eliminating'
        rv_connections: Set[int] = self._connections[rv_index]
        return sum(self._rv_log_sizes[other] for other in rv_connections)

    def weighted_fill(self, rv_index: int) -> float:
        """
        What is the total weight of fill edges are created if eliminating
        the random variable with the given index given the current state of eliminations.
        """
        assert len(self._uneliminated) > 0, 'only makes sense while eliminating'
        return self._fill_count(
            rv_index,
            self._add_sum_log2_states,
            self._divide_2,
        )

    def traditional_weighted_fill(self, rv_index: int) -> float:
        """
        What is the total traditional weight of fill edges are created if eliminating
        the random variable with the given index given the current state of eliminations.
        """
        assert len(self._uneliminated) > 0, 'only makes sense while eliminating'
        return self._fill_count(
            rv_index,
            self._add_mul_log2_states,
            self._divide_4,
        )

    def _finish_elimination(self) -> None:
        """
        All rvs are now eliminated. Do any finishing processes.
        """
        # add each rv to its own cluster
        for rv_index, cluster in enumerate(self._connections):
            cluster.add(rv_index)

        if self._maximal_clusters_only:
            # Removed subsumed clusters
            delete_sentinel: Set[int] = set()
            number_of_clusters = len(self._connections)
            for i in range(number_of_clusters):
                cluster_i = self._connections[i]
                for j in range(i + 1, number_of_clusters):
                    cluster_j = self._connections[j]
                    if cluster_i.issuperset(cluster_j):
                        # The cluster_j is a subset of cluster_i.
                        # We move cluster i to position j to preserve correct cluster order.
                        self._connections[j] = cluster_i
                        self._connections[i] = delete_sentinel
                        break
            # Remove clusters marked for deletion
            self._connections = list(filter((lambda connection: connection is not delete_sentinel), self._connections))

    def dump(self, *, prefix: str = '', indent: str = '    ') -> None:
        """
        Print a dump of the Clusters.
        This is intended for debugging and demonstration purposes.

        Args:
            prefix: optional prefix for indenting all lines.
            indent: additional prefix to use for extra indentation.
        """

        def _rv_name(_rv_idx: int) -> str:
            return repr(str(pgm.rvs[_rv_idx]))

        pgm = self._pgm

        if len(self._uneliminated) > 0:
            print(f'{prefix}Clustering incomplete.')
            print(f'{prefix}Uneliminated: ', ', '.join(_rv_name(rv_idx) for rv_idx in self._uneliminated))
            print(f'{prefix}Eliminated: ', ', '.join(_rv_name(rv_idx) for rv_idx in self._eliminated))
            print(f'{prefix}Connections:')
            for i, connections in enumerate(self._connections):
                print(f'{prefix}{indent}rv {i}:', ', '.join(_rv_name(rv_idx) for rv_idx in sorted(connections)))
            return

        print(f'{prefix}Elimination order:')
        for rv_idx in self.eliminated:
            print(f'{prefix}{indent}{_rv_name(rv_idx)}')
        print(f'{prefix}Clusters:')
        for i, cluster in enumerate(self.clusters):
            print(f'{prefix}{indent}cluster {i}:', ', '.join(_rv_name(rv_idx) for rv_idx in sorted(cluster)))

    @overload
    def _fill_count(
            self,
            rv_index: int,
            count: Callable[[int, int], float],
            finish: Callable[[float], float],
    ) -> float:
        ...

    @overload
    def _fill_count(
            self,
            rv_index: int,
            count: Callable[[int, int], int],
            finish: Callable[[int], int],
    ) -> int:
        ...

    def _fill_count(
            self,
            rv_index: int,
            fill_value: Callable[[int, int], int | float],
            result: Callable[[int | float], int | float]):
        """
        Supporting function to calculate the "fill" of a random variable.

        Args:
            rv_index: the index of the rv to compute the fill.
            fill_value: compute the fill value of two indexed random variables.
            result: compute the result value as a function of the sum of fill values.

        Returns:

        """
        fill_sum = 0
        connections: Tuple[int, ...] = tuple(self._connections[rv_index])
        for i, rv1 in enumerate(connections):
            test_connections: Set[int] = self._connections[rv1]
            for rv2 in connections[i + 1:]:
                if rv2 not in test_connections:
                    fill_sum += fill_value(rv1, rv2)
        return result(fill_sum)

    # ==============================================================
    #  The following are functions to supply to `self._fill_count`.
    # ==============================================================

    @staticmethod
    def _add_one(_1: int, _2: int) -> int:
        return 1

    def _add_sum_log2_states(self, rv1: int, rv2: int) -> float:
        return self._rv_log_sizes[rv1] + self._rv_log_sizes[rv2]

    def _add_mul_log2_states(self, rv1: int, rv2: int) -> float:
        return self._rv_log_sizes[rv1] * self._rv_log_sizes[rv2]

    @staticmethod
    def _identity(result: int) -> int:
        return result

    @staticmethod
    def _divide_2(result: float) -> float:
        return result / 2.0

    @staticmethod
    def _divide_4(result: float) -> float:
        return result / 4.0


# standard greedy algorithms

ClusterAlgorithm = Callable[[PGM], Clusters]


def min_degree(pgm: PGM) -> Clusters:
    clusters = Clusters(pgm)
    ve_greedy_min(clusters, clusters.degree)
    return clusters


def min_fill(pgm: PGM) -> Clusters:
    clusters = Clusters(pgm)
    ve_greedy_min(clusters, clusters.fill)
    return clusters


def min_degree_then_fill(pgm: PGM) -> Clusters:
    clusters = Clusters(pgm)
    ve_greedy_min(clusters, (clusters.degree, clusters.fill))
    return clusters


def min_fill_then_degree(pgm: PGM) -> Clusters:
    clusters = Clusters(pgm)
    ve_greedy_min(clusters, (clusters.fill, clusters.degree))
    return clusters


def min_weighted_degree(pgm: PGM) -> Clusters:
    clusters = Clusters(pgm)
    ve_greedy_min(clusters, clusters.weighted_degree)
    return clusters


def min_weighted_fill(pgm: PGM) -> Clusters:
    clusters = Clusters(pgm)
    ve_greedy_min(clusters, clusters.weighted_fill)
    return clusters


def min_traditional_weighted_fill(pgm: PGM) -> Clusters:
    clusters = Clusters(pgm)
    ve_greedy_min(clusters, clusters.traditional_weighted_fill)
    return clusters
