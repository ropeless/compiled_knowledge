"""
This module implements several divergences which measure the difference
between two distributions.
"""
import math
from typing import Sequence

import numpy as np

from ck.pgm import RandomVariable, rv_instances_as_indicators, PGM
from ck.probability.probability_space import ProbabilitySpace

_NAN: float = np.nan  # Not-a-number (i.e., the result of an invalid calculation).


def kl(p: ProbabilitySpace, q: ProbabilitySpace) -> float:
    """
    Compute the Kullback-Leibler divergence between p & q,
    where p is the true distribution.

    This implementation uses logarithms, base 2.

    Args:
        p: a probability space to compare to.
        q: the other probability space.

    Returns:
        the Kullback–Leibler (KL) divergence of p & q, where p is
        the true distribution.

    Raises:
        ValueError: if `p` and `q` do not have compatible random variables.specifically:
            * `len(self.rvs) == len(other.rvs)`
            * `len(other.rvs[i]) == len(self.rvs[i])` for all `i`
            * `other.rvs[i].idx == self.rvs[i].idx` for all `i`.

    Warning:
        this method will enumerate the whole probability space.
    """
    if not _compatible_rvs(p.rvs, q.rvs):
        raise ValueError('incompatible random variables')

    total = 0.0
    for x in rv_instances_as_indicators(*p.rvs):
        p_x = p.probability(*x)
        q_x = q.probability(*x)
        if p_x <= 0 or q_x <= 0:
            return _NAN
        total += p_x * math.log2(p_x / q_x)
    return total


def pseudo_kl(p: ProbabilitySpace, q: ProbabilitySpace) -> float:
    """
    A kind of KL divergence, factored by the structure of `p`.
    This is an experimental measure.

    This implementation uses logarithms, base 2.

    Args:
        p: a probability space to compare to.
        q: the other probability space.

    Returns:
        the factored histogram intersection between the two probability spaces.

    Raises:
        ValueError: if `p` and `q` do not have compatible random variables.specifically:
            * `len(self.rvs) == len(other.rvs)`
            * `len(other.rvs[i]) == len(self.rvs[i])` for all `i`
            * `other.rvs[i].idx == self.rvs[i].idx` for all `i`.
        ValueError: if not all random variable of `p` are from a single PGM, which must
            have a Bayesian network structure.
    """
    p_rvs: Sequence[RandomVariable] = p.rvs
    q_rvs: Sequence[RandomVariable] = q.rvs

    if not _compatible_rvs(p_rvs, q_rvs):
        raise ValueError('incompatible random variables')

    if len(p_rvs) == 0:
        return _NAN

    pgm: PGM = p_rvs[0].pgm
    if any(rv.pgm is not pgm for rv in p_rvs):
        raise ValueError('p random variables are not from a single PGM.')
    if not pgm.is_structure_bayesian:
        raise ValueError('p does not have Bayesian network structure.')

    # Across the two spaces, corresponding random variables are equivalent;
    # i.e., same number of states and same `idx` values. Therefore,
    # indicators from either one space can be used in both spaces.

    total: float = 0
    for factor in pgm.factors:
        for x in rv_instances_as_indicators(*factor.rvs):  # every possible state of factor rvs
            p_x = p.probability(*x)
            q_x = q.probability(*x)
            if p_x <= 0 or q_x <= 0:
                return _NAN
            total += p_x * math.log2(p_x / q_x)
    return total


def hi(p: ProbabilitySpace, q: ProbabilitySpace) -> float:
    """
    Compute the histogram intersection between this probability spaces and the given other.

    The histogram intersection between two probability spaces P and Q,
    with state spaces X, is defined as:
    ```
    HI(P, Q) = sum(min(P(x), Q(x)) for x in X)
    ```

    See:
        Swain, M.J., Ballard, D.H. Color indexing.
        International Journal of Computer Vision 7, 11–32 (1991).
        https://doi.org/10.1007/BF00130487

    Args:
        p: a probability space to compare to.
        q: the other probability space.

    Returns:
        the histogram intersection between the two probability spaces.

    Raises:
        ValueError: if `p` and `q` do not have compatible random variables.specifically:
            * `len(self.rvs) == len(other.rvs)`
            * `len(other.rvs[i]) == len(self.rvs[i])` for all `i`
            * `other.rvs[i].idx == self.rvs[i].idx` for all `i`.

    Warning:
        this method will enumerate the whole probability space.

    """
    p_rvs: Sequence[RandomVariable] = p.rvs
    q_rvs: Sequence[RandomVariable] = q.rvs

    if not _compatible_rvs(p_rvs, q_rvs):
        raise ValueError('incompatible random variables')

    # Across the two spaces, corresponding random variables are equivalent;
    # i.e., same number of states and same `idx` values. Therefore,
    # indicators from either one space can be used in both spaces.

    return sum(
        min(p.probability(*x), q.probability(*x))
        for x in rv_instances_as_indicators(*p_rvs)
    )


def fhi(p: ProbabilitySpace, q: ProbabilitySpace) -> float:
    """
    Compute the factored histogram intersection between this probability spaces and the given other.

    The factored histogram intersection between two probability spaces P and Q,
    with state spaces X and factorisation F, is defined as::

        FHI(P, Q) = 1/n sum(P(Y=y) CHI(P, Q, X | Y=y)
        where:
            CHI(P, Q, X | Y=y) = HI(P(X | Y=y), Q(X | Y=y))
            HI(P, Q) = sum(min(P(X=x), Q(X=x)) for x in f)

    The value of _n_ is the sum of P(Y=y) over all CPT rows. However,
    this always equals the number of CPTs, i.e., the number of random
    variables.

    The factorisation F is taken from the probability space `p`.

    For more information about factored histogram intersection, see the publication:
    Suresh, S., Drake, B. (2025). Sampling of Large Probabilistic Graphical Models
    Using Arithmetic Circuits. AI 2024: Advances in Artificial Intelligence. AI 2024.
    Lecture Notes in Computer Science, vol 15443. https://doi.org/10.1007/978-981-96-0351-0_13.

    Args:
        p: a probability space to compare to.
        q: the other probability space.

    Returns:
        the factored histogram intersection between the two probability spaces.

    Raises:
        ValueError: if `p` and `q` do not have compatible random variables.specifically:
            * `len(self.rvs) == len(other.rvs)`
            * `len(other.rvs[i]) == len(self.rvs[i])` for all `i`
            * `other.rvs[i].idx == self.rvs[i].idx` for all `i`.
        ValueError: if not all random variable of `p` are from a single PGM, which must
            have a Bayesian network structure.
    """
    p_rvs: Sequence[RandomVariable] = p.rvs
    q_rvs: Sequence[RandomVariable] = q.rvs

    if not _compatible_rvs(p_rvs, q_rvs):
        raise ValueError('incompatible random variables')

    if len(p_rvs) == 0:
        return 0

    pgm: PGM = p_rvs[0].pgm
    if any(rv.pgm is not pgm for rv in p_rvs):
        raise ValueError('p random variables are not from a single PGM.')
    if not pgm.is_structure_bayesian:
        raise ValueError('p does not have Bayesian network structure.')

    # Across the two spaces, corresponding random variables are equivalent;
    # i.e., same number of states and same `idx` values. Therefore,
    # indicators from either one space can be used in both spaces.

    # Loop over all CPTs, accumulating the total
    total: float = 0
    for factor in pgm.factors:
        child: RandomVariable = factor.rvs[0]
        parents: Sequence[RandomVariable] = factor.rvs[1:]
        # Loop over all rows of the CPT
        for parent_indicators in rv_instances_as_indicators(*parents):
            p_marginal = p.marginal_distribution(child, condition=parent_indicators)
            q_marginal = q.marginal_distribution(child, condition=parent_indicators)
            row_hi = np.minimum(p_marginal, q_marginal).sum().item()
            pr_row = p.probability(*parent_indicators)
            total += pr_row * row_hi

    return total / len(p_rvs)


def _compatible_rvs(rvs1: Sequence[RandomVariable], rvs2: Sequence[RandomVariable]) -> bool:
    """
    The rvs are compatible if they have the same number of random variables
    and the corresponding indicators are equal.
    """
    return (
            len(rvs1) == len(rvs2)
            and all(len(rv1) == len(rv2) and rv1.idx == rv2.idx for rv1, rv2 in zip(rvs1, rvs2))
    )
