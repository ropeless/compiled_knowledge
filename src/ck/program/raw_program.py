from dataclasses import dataclass
from typing import Callable, Sequence, TypeAlias

import numpy as np
import ctypes as ct


from ck.utils.np_extras import NDArrayNumeric, DTypeNumeric

RawProgramFunction: TypeAlias = Callable[[ct.POINTER, ct.POINTER, ct.POINTER], None]
"""
RawProgramFunction is a function of three ctypes arrays, returning nothing.
Args:
    [0]: input  values,
    [1]: temporary working memory,
    [2]: output values.
"""


@dataclass
class RawProgram:
    """
    A raw program is returned by a circuit compiler to provide execution of
    the function defined by a compiled circuit.

    A `RawProgram` is a `Callable`; given an array of input variable values,
    return a numpy array of result values. Calling a RawProgram is not necessarily
    an efficient method for executing a program as buffers are reallocated for
    each call. Alternatively, a `RawProgram` can be wrapped in a `ProgramBuffer`
    for computationally efficient memory buffer reuse.
    """

    function: RawProgramFunction
    """a function of three ctypes arrays, returning nothing."""

    dtype: DTypeNumeric
    """the numpy data type of  the array values."""

    number_of_vars: int
    """the number of input values (first function argument)."""

    number_of_tmps: int
    """the number of working memory values (second function argument)."""

    number_of_results: int
    """the number of result values (third function argument)."""

    var_indices: Sequence[int]
    """
    a map from the index of inputs (from 0 to self.number_of_vars - 1) to the index
    of the corresponding circuit var.
    """

    def __call__(self, var_values: NDArrayNumeric | Sequence[int | float] | int | float) -> NDArrayNumeric:
        """
        Call the raw program as a function. This method will allocate numpy arrays of type `self.dtype`
        for input, temporary, and output values. If `var_values` is a numpy array of the needed
        dtype then it will be used directly.

        Args:
            var_values: the input variable values. This can be a numpy array, a Python sequence of
                floats or int, or a single float or int. The number of input values must equal
                `self.number_of_vars`.

        Returns:
            a numpy array of result values with shape `(self.number_of_results,)`.
        """
        array_vars: NDArrayNumeric
        if isinstance(vars, np.ndarray):
            if var_values.dtype != self.dtype:
                array_vars = var_values.astype(self.dtype)
            else:
                array_vars = self.dtype
        elif isinstance(vars, (int, float)):
            array_vars = np.array([var_values], dtype=self.dtype)
        else:
            array_vars = np.array(var_values, dtype=self.dtype)

        if array_vars.shape != (self.number_of_vars,):
            raise ValueError(f'input array incorrect shape: got {array_vars.shape} expected ({self.number_of_vars},)')

        array_tmps: NDArrayNumeric = np.zeros(self.number_of_tmps, dtype=self.dtype)
        array_outs: NDArrayNumeric = np.zeros(self.number_of_results, dtype=self.dtype)

        ptr_type = ct.POINTER(np.ctypeslib.as_ctypes_type(self.dtype))
        c_array_vars = array_vars.ctypes.data_as(ptr_type)
        c_array_tmps = array_tmps.ctypes.data_as(ptr_type)
        c_array_outs = array_outs.ctypes.data_as(ptr_type)

        self.function(c_array_vars, c_array_tmps, c_array_outs)
        return array_outs

    def dump(self, *, prefix: str = '', indent: str = '  ') -> None:
        """
        Print a dump of the PGM.
        This is intended for demonstration and debugging purposes.

        Args:
            prefix: optional prefix for indenting all lines.
            indent: additional prefix to use for extra indentation.
        """
        print(f'{prefix}{self.__class__.__name__}')
        print(f'{prefix}signature = [{self.number_of_vars}] -> [{self.number_of_results}]')
        print(f'{prefix}temps = {self.number_of_tmps}')
        print(f'{prefix}dtype = {self.dtype}')
        print(f'{prefix}var_indices = {self.var_indices}')
