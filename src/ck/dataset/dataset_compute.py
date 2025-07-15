"""
A collection of functions to compute values over datasets using programs.
"""
import ctypes as ct
from typing import Optional, List, Dict

import numpy as np

from ck.dataset import SoftDataset
from ck.pgm import Indicator, RandomVariable
from ck.pgm_circuit.slot_map import SlotMap
from ck.program import RawProgram
from ck.utils.np_extras import NDArray, NDArrayNumeric


def accumulate_compute(
        program: RawProgram,
        slot_arrays: NDArray,
        *,
        weights: Optional[NDArray] = None,
        accumulator: Optional[NDArray] = None,
) -> NDArray:
    """
    Apply the given program to every instance in the dataset, summing all results over the instances.

    Args:
        program: the mathematical transformation to apply to the data.
        slot_arrays: a 2D numpy array of shape (number_of_instances, number_of_slots). Appropriate
            slot arrays can be constructed from a soft dataset using `get_slot_arrays`.
        weights: and optional 1D array of instance weights, of shape (number_of_instances, ), and
            co-indexed with slot_arrays.
        accumulator: an optional array to perform the result accumulation, summing with the initial
            values of the provided accumulator.

    Returns:
        total_weight, accumulator

    Raises:
        ValueError: if slot_arrays.shape is not `(..., program.number_of_vars)`.
        ValueError: if an accumulator is provided, but is not shape `(program.number_of_results, )`.
        ValueError: if weights provided, but is not shape `(slot_arrays.shape[0],)`.
    """
    number_of_results: int = program.number_of_results
    number_of_vars: int = program.number_of_vars

    if len(slot_arrays.shape) != 2 or slot_arrays.shape[1] != program.number_of_vars:
        raise ValueError(f'slot arrays expected shape (..., {number_of_vars}) but got {slot_arrays.shape}')

    if accumulator is None:
        accumulator = np.zeros(number_of_results, dtype=program.dtype)
    elif accumulator.shape != (number_of_results,):
        raise ValueError(f'accumulator shape {accumulator.shape} does not match number of results: {number_of_results}')

    if slot_arrays.dtype != program.dtype:
        raise ValueError(f'slot arrays dtype {slot_arrays.dtype} does not match program.dtype: {program.dtype}')
    if accumulator.dtype != program.dtype:
        raise ValueError(f'accumulator dtype {slot_arrays.dtype} does not match program.dtype: {program.dtype}')

    ptr_type = ct.POINTER(np.ctypeslib.as_ctypes_type(program.dtype))

    # Create buffers for program function tmps and outputs
    # We do not need to create a buffer for program function inputs as that
    # will be provided by `slot_arrays`.
    array_outs: NDArrayNumeric = np.zeros(program.number_of_results, dtype=program.dtype)
    array_tmps: NDArrayNumeric = np.zeros(program.number_of_tmps, dtype=program.dtype)
    c_array_tmps = array_tmps.ctypes.data_as(ptr_type)
    c_array_outs = array_outs.ctypes.data_as(ptr_type)

    if weights is None:
        # This is the unweighed version
        for instance in slot_arrays:
            c_array_vars = instance.ctypes.data_as(ptr_type)
            program.function(c_array_vars, c_array_tmps, c_array_outs)
            accumulator += array_outs

    else:
        # This is the weighed version
        expected_shape = (slot_arrays.shape[0],)
        if weights.shape != expected_shape:
            raise ValueError(f'weight shape {weights.shape} is not as expected : {expected_shape}')

        for weight, instance in zip(weights, slot_arrays):
            c_array_vars = instance.ctypes.data_as(ptr_type)
            program.function(c_array_vars, c_array_tmps, c_array_outs)
            accumulator += array_outs * weight

    return accumulator


def get_slot_arrays(
        dataset: SoftDataset,
        number_of_slots: int,
        slot_map: SlotMap,
) -> NDArray:
    """
    For each slot from 0 to number_of_slots - 1, get the 1D vector
    from the dataset that can be used to set each slot.

    This function can be used to prepare slot arrays for `accumulate_compute`.

    Returns:
        a 2D numpy array of shape (len(dataset), number_of_slots),

    Raises:
        ValueError: if multiple indicators for a slot in the slot map
        ValueError: if there are slots with no indicator in slot map
    """

    # Special case, no slots
    # We treat this specially to ensure the right shape of the result
    if number_of_slots == 0:
        return np.empty(shape=(len(dataset), 0))

    # Use the slot map to work out which indicator corresponds to each slot.
    indicators: List[Optional[Indicator]] = [None] * number_of_slots
    for indicator, slot in slot_map.items():
        if 0 <= slot < number_of_slots and indicator is not None:
            if indicators[slot] is not None and indicators[slot] != indicator:
                raise ValueError(f'multiple indicators for slot: {slot}')
            indicators[slot] = indicator
    missing_slots = [i for i, indicator in enumerate(indicators) if indicator is None]
    if len(missing_slots) > 0:
        missing_slots_str = ', '.join(str(slot) for slot in missing_slots)
        raise ValueError(f'slots with no indicator in slot map: {missing_slots_str}')

    # Map rv index to state_weights of the dataset
    rv: RandomVariable
    state_weights: Dict[int, NDArray] = {
        rv.idx: dataset.state_weights(rv)
        for rv in dataset.rvs
    }

    # Get the columns of the resulting matrix
    columns = [
        state_weights[indicator.rv_idx][:, indicator.state_idx]
        for indicator in indicators
    ]

    # Concatenate the columns into a matrix
    return np.column_stack(columns)
