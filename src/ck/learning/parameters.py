"""
General functions for setting the parameter values of a PGM.
"""
from typing import List, Tuple, TypeAlias

import numpy as np

from ck.dataset.cross_table import CrossTable
from ck.pgm import PGM, CPTPotentialFunction, Instance, SparsePotentialFunction, DensePotentialFunction, Factor
from ck.utils.map_list import MapList
from ck.utils.np_extras import NDArrayFloat64


ParameterValues: TypeAlias = List[CrossTable]


def make_factors(pgm: PGM, parameter_values: List[CrossTable]) -> None:
    for factor in parameter_values:
        pgm.new_factor(*factor.rvs)
    set_potential_functions(pgm, parameter_values)


def set_potential_functions(pgm: PGM, parameter_values: List[CrossTable]) -> None:
    """
    Set the potential function of each PGM factor to one heuristically chosen,
    using the given parameter values. Then set the parameter values of the potential
    function to those given by `parameter_values`.

    This function modifies `pgm` in-place, iteratively calling `set_potential_function`.

    Args:
        pgm (PGM): the PGM to have its potential functions set.
        parameter_values: the parameter values,
    """
    for factor, factor_parameter_values in zip(pgm.factors, parameter_values):
        set_potential_function(factor, factor_parameter_values)


def set_potential_function(factor: Factor, parameter_values: CrossTable) -> None:
    """
    Set the potential function of the given factor to one heuristically chosen,
    using the given parameter values. Then set the parameter values of the potential
    function to those given by `parameter_values`.

    The potential function will be either a ZeroPotentialFunction, DensePotentialFunction,
    or SparsePotentialFunction.

    This function modifies `factor` in-place.

    Args:
        factor: The factor to update.
        parameter_values: the parameter values,
    """
    number_of_parameters: int = len(parameter_values)
    if number_of_parameters == 0:
        factor.set_zero()
    else:
        if number_of_parameters < 100 or number_of_parameters > factor.number_of_states * 0.9:
            pot_function: DensePotentialFunction = factor.set_dense()
        else:
            pot_function: SparsePotentialFunction = factor.set_sparse()
        for instance, weight in parameter_values.items():
            pot_function[instance] = weight


def set_zero(pgm: PGM) -> None:
    """
    Set the potential function of each PGM factor to zero.
    """
    for factor in pgm.factors:
        factor.set_zero()


def set_dense(pgm: PGM, parameter_values: List[CrossTable]) -> None:
    """
    Set the potential function of each PGM factor to a DensePotentialFunction,
    using the given parameter values.
    """
    for factor, cpt in zip(pgm.factors, parameter_values):
        pot_function: DensePotentialFunction = factor.set_dense()
        for instance, weight in cpt.items():
            pot_function[instance] = weight


def set_sparse(pgm: PGM, parameter_values: List[CrossTable]) -> None:
    """
    Set the potential function of each PGM factor to a SparsePotentialFunction,
    using the given parameter values.
    """
    for factor, cpt in zip(pgm.factors, parameter_values):
        pot_function: SparsePotentialFunction = factor.set_sparse()
        for instance, weight in cpt.items():
            pot_function[instance] = weight


def set_cpt(pgm: PGM, parameter_values: List[CrossTable], normalise_cpds: bool = True) -> None:
    """
    Set the potential function of each PGM factor to a CPTPotentialFunction,
    using the given parameter values.
    """
    for factor, cpt in zip(pgm.factors, parameter_values):
        pot_function: CPTPotentialFunction = factor.set_cpt()

        # Group cpt values by parent instance
        cpds: MapList[Instance, Tuple[int, float]] = MapList()
        for instance, weight in cpt.items():
            cpds.append(instance[1:], (instance[0], weight))

        # Set the CPDs
        cpd_size = len(cpt.rvs[0])  # size of the child random variable
        for parent_instance, cpd in cpds.items():
            cpd_array: NDArrayFloat64 = np.zeros(cpd_size, dtype=np.float64)
            for child_state_index, weight in cpd:
                cpd_array[child_state_index] = weight
            if normalise_cpds:
                cpd_array /= cpd_array.sum()
            pot_function.set_cpd(parent_instance, cpd_array)
