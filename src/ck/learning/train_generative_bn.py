from __future__ import annotations

from typing import List, Mapping, Tuple

from ck.dataset import SoftDataset, HardDataset
from ck.dataset.cross_table import CrossTable, cross_table_from_dataset
from ck.learning.parameters import set_potential_functions, ParameterValues
from ck.pgm import PGM


def train_generative_bn(
        pgm: PGM,
        dataset: HardDataset | SoftDataset,
        *,
        dirichlet_prior: float | Mapping[int, float | CrossTable] = 0,
        check_bayesian_network: bool = True,
) -> None:
    """
    Maximum-likelihood, generative training for a Bayesian network.

    The potential function of the given PGM will be set to new potential functions
    with the learned parameter values.

    Args:
        pgm: the probabilistic graphical model defining the model structure.
            Potential function values are ignored and need not be set.
        dataset: a dataset of random variable states.
        dirichlet_prior: provides a Dirichlet prior for each factor in `pgm`.
            This can be represented in multiple ways:
            (a) as a uniform prior that is the same for all factors, represented as a float value,
            (b) as a mapping from a factor index to a uniform prior, i.e., a float value,
            (c) as a mapping from a factor index to an arbitrary Dirichlet prior, i.e., a cross-table.
            If there is no entry in the mapping for a factor, then the value 0 will be used for that factor.
            If a cross-table is provided as a prior, then it must have the same random variables as
            the factor it pertains to.
            The default value for `dirichlet_prior` is 0.
            See `CrossTable` for more explanation.
        check_bayesian_network: if true and not `pgm.is_structure_bayesian` an exception will be raised.

    Raises:
        ValueError: if the given PGM does not have a Bayesian network structure, and check_bayesian_network is True.
    """
    if check_bayesian_network and not pgm.is_structure_bayesian:
        raise ValueError('the given PGM is not a Bayesian network')
    cpts: List[CrossTable] = get_cpts(
        pgm=pgm,
        dataset=dataset,
        dirichlet_prior=dirichlet_prior,
    )
    set_potential_functions(pgm, cpts)


def get_cpts(
        pgm: PGM,
        dataset: HardDataset | SoftDataset,
        *,
        dirichlet_prior: float | Mapping[int, float | CrossTable] = 0,
) -> ParameterValues:
    """
    This function applies `cpt_from_crosstab` to each cross-table from `get_factor_cross_tables`.
    The resulting parameter values are CPTs that can be used directly to update the parameters
    of the given PGM, so long as it has a Bayesian network structure.

    To update the given PGM from the resulting `cpts` use `set_potential_functions(pgm, cpts)`.

    Args:
        pgm: the probabilistic graphical model defining the model structure.
            Potential function values are ignored and need not be set.
        dataset: a dataset of random variable states.
        dirichlet_prior: provides a Dirichlet prior for each factor in `pgm`.
            This can be represented in multiple ways:
            (a) as a uniform prior that is the same for all factors, represented as a float value,
            (b) as a mapping from a factor index to a uniform prior, i.e., a float value,
            (c) as a mapping from a factor index to an arbitrary Dirichlet prior, i.e., a cross-table.
            If there is no entry in the mapping for a factor, then the value 0 will be used for that factor.
            If a cross-table is provided as a prior, then it must have the same random variables as
            the factor it pertains to.
            The default value for `dirichlet_prior` is 0.
            See `CrossTable` for more explanation.

    Returns:
        ParameterValues object, a CPT for each factor in the given PGM, as a list of cross-tables, co-indexed
        with the PGM factors.
    """
    cross_tables: List[CrossTable] = get_factor_cross_tables(
        pgm=pgm,
        dataset=dataset,
        dirichlet_prior=dirichlet_prior,
    )
    cpts: List[CrossTable] = list(map(cpt_from_crosstab, cross_tables))
    return cpts


def get_factor_cross_tables(
        pgm: PGM,
        dataset: HardDataset | SoftDataset,
        *,
        dirichlet_prior: float | Mapping[int, float | CrossTable] = 0,
) -> ParameterValues:
    """
    Compute a cross-table for each factor of the given PGM, using the data from
    the given dataset.

    Args:
        pgm: the probabilistic graphical model defining the model structure.
            Potential function values are ignored and need not be set.
        dataset: a dataset of random variable states.
        dirichlet_prior: provides a Dirichlet prior for each factor in `pgm`.
            This can be represented in multiple ways:
            (a) as a uniform prior that is the same for all factors, represented as a float value,
            (b) as a mapping from a factor index to a uniform prior, i.e., a float value,
            (c) as a mapping from a factor index to an arbitrary Dirichlet prior, i.e., a cross-table.
            If there is no entry in the mapping for a factor, then the value 0 will be used for that factor.
            If a cross-table is provided as a prior, then it must have the same random variables as
            the factor it pertains to.
            The default value for `dirichlet_prior` is 0.
            See `CrossTable` for more explanation.

    Returns:
        ParameterValues object, a crosstable for each factor in the given PGM, as
        per `cross_table_from_dataset`.

    Assumes:
        every random variable of the PGM is in the dataset.
    """
    factor_dict: Mapping[int, float | CrossTable]
    default_prior: float
    if isinstance(dirichlet_prior, (float, int)):
        factor_dict = {}
        default_prior = dirichlet_prior
    else:
        factor_dict = dirichlet_prior
        default_prior = 0

    cross_tables: List[CrossTable] = [
        cross_table_from_dataset(
            dataset,
            factor.rvs,
            dirichlet_prior=factor_dict.get(factor.idx, default_prior),
        )
        for factor in pgm.factors
    ]
    return cross_tables


def cpt_from_crosstab(crosstab: CrossTable) -> CrossTable:
    """
    Convert the given cross-table to a conditional probability table (CPT),
    assuming the first random variable of the cross-table is the child
    and remaining random variables are the parents.

    Args:
        crosstab: a CrossTable representing the weight of unique instances.

    Returns:
        A cross-table that is a conditional probability table.

    Assumes:
        the first random variable in `crosstab.rvs` is the child random variable.
    """
    return cpt_and_parent_sums_from_crosstab(crosstab)[0]


def cpt_and_parent_sums_from_crosstab(crosstab: CrossTable) -> Tuple[CrossTable, CrossTable]:
    """
    Convert the given cross-table to a conditional probability table (CPT),
    assuming the first random variable of the cross-table is the child
    and remaining random variables are the parents.

    Args:
        crosstab: a CrossTable representing the weight of unique instances.

    Returns:
        A cross-table that is a conditional probability table.
        A cross-table of the parent sums that were divided out of `crosstab`

    Assumes:
        the first random variable in `crosstab.rvs` is the child random variable.
    """
    # Get the sum of weights for parent states
    parent_sums: CrossTable = CrossTable(
        rvs=crosstab.rvs[1:],
        update=(
            (instance[1:], weight)
            for instance, weight in crosstab.items()
        )
    )

    # Construct the normalised cross-tables, i.e., the CPTs.
    cpt = CrossTable(
        rvs=crosstab.rvs,
        update=(
            (instance, weight / parent_sums[instance[1:]])
            for instance, weight in crosstab.items()
        )
    )

    return cpt, parent_sums
