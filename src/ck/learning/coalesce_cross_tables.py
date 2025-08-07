from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Sequence, Dict

import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import lsqr

from ck.dataset.cross_table import CrossTable
from ck.pgm import RandomVariable, Instance
from ck.utils.iter_extras import combos
from ck.utils.np_extras import NDArrayFloat64


def coalesce_cross_tables(crosstabs: Sequence[CrossTable], rvs: Sequence[RandomVariable]) -> CrossTable:
    """
    Rationalise multiple cross-tables into a single cross-table.

    This method implements a solution finding the best vector `a`
    that solves `b = m a`, subject to `a[i] >= 0`.

    `a` is a column vector with one entry for each instance of rvs seen in the cross-tables.

    `b` is a column vector containing all probabilities from all cross-tables.

    `m` is a sparse matrix with m[i, j] = 1 where b[j] is in the sum for source probability a[i].

    "Best" means the vector `a` with:
    * `a[i] >= 0` for all i, then
    * `b - m a` having the smallest L2 norm, then
    * `a` having the smallest L2 norm.

    The given crosstables will be used to form `b`. The entries each cross-table will be normalised
    to represent a probability distribution over the cross-table's instances (keys).

    Args:
        crosstabs: a collection of cross-tables to coalesce.
        rvs: the random variables that cross-tables will be projected on to.

    Returns:
        a cross-table defined for the given `rvs`, with values inferred from `crosstabs`
    """
    if len(crosstabs) == 0:
        return CrossTable(rvs)

    m: dok_matrix
    b: np.ndarray
    a_keys: Sequence[Instance]
    m, b, a_keys = _make_matrix(crosstabs, rvs)

    a = _solve(m, b)

    return CrossTable(
        rvs,
        update=(
            (instance, weight)
            for instance, weight in zip(a_keys, a)
        ),
    )


def _solve(m: dok_matrix, b: np.ndarray) -> np.ndarray:
    """
    Find the best 'a' for `b = m a` subject to `a[i] >= 0`.
    """
    assert len(b.shape) == 1, 'b should be a vector'
    assert b.shape[0] == m.shape[0], 'b and m must be compatible'

    return _solve_sam(m, b)
    # return _solve_lsqr(m, b)
    # return _solve_pulp_l1(m, b)


def _solve_sam(m: dok_matrix, b: np.ndarray) -> np.ndarray:
    """
    Find the best 'a' for `b = m a` subject to `a[i] >= 0`.

    Uses a custom 'split and mean' (SAM) method.
    """
    sam = _SAM(m, b)
    a, error = sam.solve(
        max_iterations=100,
        tolerance=1e-6,
        change_tolerance=1e-12
    )
    return a


def _solve_lsqr(m: dok_matrix, b: np.ndarray) -> np.ndarray:
    """
    Find the best 'a' for `b = m a` subject to `a[i] >= 0`.

    Uses scipy `lsqr` method, with a heuristic to fix negative values in `a`.
    """
    a: np.ndarray
    # Pycharm type checker incorrectly infers the type signature of `lsqr`
    # noinspection PyTypeChecker
    a, istop, itn, r1norm, r2norm, _, _, _, _, _ = lsqr(m, b)

    # Negative values or values > 1 are not a valid solution.

    # Heuristic fix up...
    if len(a) > 0:
        min_val = np.min(a)
        if min_val < 0:
            # We could just let the negative values get truncated to zero, but
            # empirically the results seem better when we shift all parameters up.
            a[:] -= min_val

    return a


# This approach is unsatisfactory as we should minimise the L2 norm
# rather than the L1 norm.
#
# def _solve_pulp_l1(m: dok_matrix, b: np.ndarray) -> np.ndarray:
#     """
#     Find the best 'a' for `b = m a` subject to `a[i] >= 0`.
#
#     Uses pulp LpProblem to minimise the L1 norm of `a`.
#
#     This method will only work if there is an exact solution.
#     If not, then we call _solve_sam as a fallback.
#     """
#     import pulp
#
#     a_size = m.shape[1]
#     b_size = b.shape[0]
#
#     prob = pulp.LpProblem('solver', pulp.LpMinimize)
#     x = [pulp.LpVariable(f'x{i}', lowBound=0) for i in range(a_size)]
#
#     # The objective: minimise the L1 norm of x.
#     # The sum(x) is the L1 norm because each element of x is constrained >= 0.
#     prob.setObjective(pulp.lpSum(x))
#
#     # The constraints
#     constraints = [pulp.LpAffineExpression() for _ in range(b_size)]
#     for row, col in m.keys():
#         constraints[row].addterm(x[col], 1)
#     for c, b_i in zip(constraints, b):
#         prob.addConstraint(c == b_i)
#
#     _PULP_TIMEOUT = 60  # seconds
#     status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=_PULP_TIMEOUT))
#
#     if status == pulp.LpStatusOptimal:
#         return np.array([pulp.value(x_var) for x_var in x])
#     else:
#         return _solve_sam(m, b)


def _sum_out_unneeded_rvs(crosstab: CrossTable, rvs: Sequence[RandomVariable]) -> CrossTable:
    """
    Project the given cross-table as needed to ensure all random
    variables in the result are in `rvs`.
    """
    available_rvs = set(crosstab.rvs)
    project_rvs = available_rvs.intersection(rvs)
    if len(project_rvs) == len(available_rvs):
        # No projection is required
        return crosstab
    else:
        return crosstab.project(list(project_rvs))


def _make_matrix(
        crosstabs: Sequence[CrossTable],
        rvs: Sequence[RandomVariable],
) -> Tuple[dok_matrix, np.ndarray, Sequence[Instance]]:
    """
    Create the `m` matrix and `b` vector for solving `b = m a`.

    Args:
        crosstabs: a collection of cross-tables to coalesce.
        rvs: the random variables that cross-tables will be projected on to.

    Returns:
        the tuple (m, b, a_keys) where
        'm' is a sparse matrix,
        'b' is a numpy array of crosstab probabilities (normalised as needed),
        'a_keys' are the keys for the solution probabilities, co-indexed with `a`.
    """

    # Sum out any unneeded random variables
    crosstabs: Sequence[CrossTable] = [
        _sum_out_unneeded_rvs(crosstab, rvs)
        for crosstab in crosstabs
    ]

    m_cols: Dict[Instance, _MCol] = {}
    b_list: List[float] = []
    a_keys: List[Instance] = []

    rv_index: Dict[RandomVariable, int] = {rv: i for i, rv in enumerate(rvs)}

    # instance_template[i] is a list of the possible states of rvs[i]
    instance_template: List[List[int]] = [list(range(len(rv))) for rv in enumerate(rvs)]

    for crosstab in crosstabs:

        # get `to_rv` such that crosstab.rvs[i] = rvs[to_rv[i]]
        to_rv = [rv_index.get(rv) for rv in crosstab.rvs]

        # Make instance_options which is a clone of instance_template but with
        # a singleton list replacing the rvs that this crosstab has.
        # For now the state in each singleton is set to -1, however, later
        # they will be set to the actual states of instances in the current crosstab.
        instance_options = list(instance_template)
        for i in to_rv:
            instance_options[i] = [-1]

        total = crosstab.total_weight()
        for crosstab_instance, weight in crosstab.items():

            # Work out what instances get summed to create the crosstab_instance weight.
            # This just overrides the singleton states of `instance_options` with the
            # actual state of the crosstab_instance.
            for state, i in zip(crosstab_instance, to_rv):
                instance_options[i][0] = state

            # Grow the b list with our instance probability
            b_i = len(b_list)
            b_list.append(weight / total)

            # Iterate over all states of `rvs` that matches the current crosstab_instance
            # recording `b_i` in the column for those matching instances.
            for instance in combos(instance_options):
                m_col = m_cols.get(instance)
                if m_col is None:
                    m_cols[instance] = _MCol(instance, len(a_keys), [b_i])
                    a_keys.append(instance)
                else:
                    m_col.col.append(b_i)

    # Construct the m matrix from m_cols
    m = dok_matrix((len(b_list), len(a_keys)), dtype=np.double)
    for m_col in m_cols.values():
        j = m_col.column_index
        for i in m_col.col:
            m[i, j] = 1

    # Construct the b vector
    b = np.array(b_list, dtype=np.double)

    return m, b, a_keys


@dataclass
class _MCol:
    key: Instance
    column_index: int
    col: List[int]


@dataclass
class _SM:
    split: float
    a: float

    def diff(self) -> float:
        return self.split - self.a


class _SAM:
    """
    Split and Mean method for finding 'a'
    in b = m a
    subject to a[i] >= 0.

    Assumes all elements of `m` are either zero or one.
    """

    def __init__(self, m: dok_matrix, b: NDArrayFloat64, use_lsqr: bool = True):
        """
        Allocate the memory required for a SAM solver.

        Args:
            m: the summation matrix
            b: the vector of resulting probabilities
            use_lsqr: whether to use LSQR or not to initialise the solution.
        """
        # Replicate the sparse m matrix, as a list of lists of _SM objects,
        # where we have both row major and column major representations.
        a_size: int = m.shape[1]
        b_size: int = m.shape[0]
        a_idx: List[List[_SM]] = [[] for _ in range(a_size)]
        b_idx: List[List[_SM]] = [[] for _ in range(b_size)]
        for (i, j), m_val in m.items():
            if m_val != 0:
                sm = _SM(0, 0)
                a_idx[j].append(sm)
                b_idx[i].append(sm)

        self._a: NDArrayFloat64 = np.zeros(a_size, dtype=np.double)
        self._b: NDArrayFloat64 = b
        self._m: dok_matrix = m
        self._a_idx: List[List[_SM]] = a_idx
        self._b_idx: List[List[_SM]] = b_idx
        self._use_lsqr = use_lsqr

    def solve(self, max_iterations: int, tolerance: float, change_tolerance: float) -> Tuple[np.ndarray, float]:
        """
        Initialize split values then iterate (mean step, split step).

        Args:
            max_iterations: maximum number of iterations.
            tolerance: terminate iterations if error <= tolerance.
            change_tolerance: terminate iterations if change in error <= change_tolerance.

        Returns:
            tuple ('a', 'error') where 'a' is the current solution after this step
            and 'error' is the sum of absolute errors.
        """
        self._initialize_split_values()
        iteration = 0
        prev_error = 0
        while True:
            iteration += 1
            a, error = self._mean_step()
            if error <= tolerance or abs(error - prev_error) <= change_tolerance or iteration >= max_iterations:
                return a, error
            prev_error = error
            self._split_step()

    def _initialize_split_values(self):
        """
        Take each 'b' value and split it across its SM cells.

        If 'self._use_lsqr' is True then the split is based on a solution
        using scipy lsqr, otherwise the split is even for each 'b' value.
        """
        for b_val, b_list in zip(self._b, self._b_idx):
            len_b_list = len(b_list)
            if len_b_list > 0:
                split_val = b_val / len_b_list
                for sm in b_list:
                    sm.split = split_val

        if self._use_lsqr:
            a = _solve_lsqr(self._m, self._b)
            assert len(a) == len(self._a)
            for a_val, a_list in zip(a, self._a_idx):
                for sm in a_list:
                    sm.a = a_val
            self._split_step()

    def _mean_step(self) -> Tuple[np.ndarray, float]:
        """
        Take the current split values to determine the 'a' values
        as the mean across relevant SM cells.

        Assumes the previous step was either 'initialize_split_values'
        or a 'split step'.

        Returns:
            tuple ('a', 'error') where 'a' is the current solution after this step
            and 'error' is the sum of absolute errors.
        """
        error = 0.0
        a = self._a
        for i, a_list in enumerate(self._a_idx):
            sum_val = sum(sm.split for sm in a_list)
            a_val = sum_val / len(a_list)
            a[i] = a_val
            for sm in a_list:
                sm.a = a_val
                error += abs(sm.diff())
        return a, error

    def _split_step(self):
        """
        Take the difference between the split 'b' values and current 'a' values to
        redistribute the split 'b' values.

        Assumes the previous step was a 'mean step'.
        """
        for b_val, b_list in zip(self._b, self._b_idx):
            if len(b_list) <= 1:
                # Too small to split
                continue
            pos = 0
            neg = 0
            for sm in b_list:
                diff = sm.diff()
                if diff >= 0:
                    pos += diff
                else:
                    neg -= diff
            mass = min(pos, neg)
            if mass == 0:
                # No mass to redistribute
                continue
            pos = mass / pos
            neg = mass / neg
            for sm in b_list:
                diff = sm.diff()
                if diff >= 0:
                    mass = diff * pos
                else:
                    mass = diff * neg
                sm.split -= mass
