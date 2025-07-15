from dataclasses import dataclass
from typing import Dict, Tuple, List

import numpy as np

from ck.dataset import SoftDataset, HardDataset
from ck.dataset.cross_table import CrossTable, cross_table_from_dataset
from ck.pgm import PGM, Instance, DensePotentialFunction, Shape, natural_key_idx, SparsePotentialFunction
from ck.utils.iter_extras import multiply
from ck.utils.np_extras import NDArrayFloat64


@dataclass
class ParameterValues:
    """
    A ParameterValues object represents learned parameter values of a PGM.
    """
    pgm: PGM
    """
    The PGM that the parameter values pertains to.
    """

    cpts: List[Dict[Instance, NDArrayFloat64]]
    """
    A list of CPTs co-indexed with `pgm.factors`. Each CPT is a dict
    mapping from instances of the parent random variables (of the factors)
    to the child conditional probability distribution (CPD).
    """

    def set_zero(self) -> None:
        """
        Set the potential function of each PGM factor to zero.
        """
        for factor in self.pgm.factors:
            factor.set_zero()

    def set_cpt(self) -> None:
        """
        Set the potential function of each PGM factor to a CPTPotentialFunction,
        using our parameter values.
        """
        for factor, cpt in zip(self.pgm.factors, self.cpts):
            factor.set_cpt().set(*cpt.items())

    def set_dense(self) -> None:
        """
        Set the potential function of each PGM factor to a DensePotentialFunction,
        using our parameter values.
        """
        for factor, cpt in zip(self.pgm.factors, self.cpts):
            pot_function: DensePotentialFunction = factor.set_dense()
            parent_shape: Shape = factor.shape[1:]
            child_state: int
            value: float
            if len(parent_shape) == 0:
                cpd: NDArrayFloat64 = cpt[()]
                for child_state, value in enumerate(cpd):
                    pot_function[child_state] = value
            else:
                parent_space: int = multiply(parent_shape)
                parent_states: Instance
                cpd: NDArrayFloat64
                for parent_states, cpd in cpt.items():
                    idx: int = natural_key_idx(parent_shape, parent_states)
                    for value in cpd:
                        pot_function[idx] = value
                        idx += parent_space

    def set_sparse(self) -> None:
        """
        Set the potential function of each PGM factor to a SparsePotentialFunction,
        using our parameter values.
        """
        for factor, cpt in zip(self.pgm.factors, self.cpts):
            pot_function: SparsePotentialFunction = factor.set_sparse()
            parent_states: Instance
            child_state: int
            cpd: NDArrayFloat64
            value: float
            for parent_states, cpd in cpt.items():
                for child_state, value in enumerate(cpd):
                    key = (child_state,) + parent_states
                    pot_function[key] = value


def train_generative_bn(
        pgm: PGM,
        dataset: HardDataset | SoftDataset,
        *,
        dirichlet_prior: float = 0,
        check_bayesian_network: bool = True,
) -> ParameterValues:
    """
    Maximum-likelihood, generative training for a Bayesian network.

    Args:
        pgm: the probabilistic graphical model defining the model structure.
            Potential function values are ignored and need not be set.
        dataset: a dataset of random variable states.
        dirichlet_prior: a real number >= 0. See `CrossTable` for an explanation.
        check_bayesian_network: if true and not pgm.is_structure_bayesian an exception will be raised.

    Returns:
        a  ParameterValues object that can be used to update the parameters of the given PGM.

    Raises:
        ValueError: if the given PGM does not have a Bayesian network structure, and check_bayesian_network is True.
    """
    if check_bayesian_network and not pgm.is_structure_bayesian:
        raise ValueError('the given PGM is not a Bayesian network')
    cpts: List[Dict[Instance, NDArrayFloat64]] = [
        cpt_from_crosstab(cross_table_from_dataset(dataset, factor.rvs, dirichlet_prior=dirichlet_prior))
        for factor in pgm.factors
    ]
    return ParameterValues(pgm, cpts)


def cpt_from_crosstab(crosstab: CrossTable) -> Dict[Instance, NDArrayFloat64]:
    """
    Make a conditional probability table (CPT) from a cross-table.

    Args:
        crosstab: a CrossTable representing the weight of unique instances.

    Returns:
        a mapping from instances of the parent random variables to the child
        conditional probability distribution (CPD).

    Assumes:
        the first random variable in `crosstab.rvs` is the child random variable.
    """
    # Number of states for the child random variable.
    child_size: int = len(crosstab.rvs[0])

    # Get distribution over child states for seen parent states
    parents_weights: Dict[Instance, NDArrayFloat64] = {}
    for state, weight in crosstab.items():
        parent_state: Tuple[int, ...] = state[1:]
        child_state: int = state[0]
        parent_weights = parents_weights.get(parent_state)
        if parent_weights is None:
            parents_weights[parent_state] = parent_weights = np.zeros(child_size, dtype=np.float64)
        parent_weights[child_state] += weight

    # Normalise
    for parent_state, parent_weights in parents_weights.items():
        parent_weights /= parent_weights.sum()

    return parents_weights
