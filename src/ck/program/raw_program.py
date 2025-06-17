from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import ctypes as ct


from ck.utils.np_extras import NDArrayNumeric, DTypeNumeric

# RawProgramFunction is a function of three ctypes arrays, returning nothing.
# Args:
#     [0]: input  values,
#     [1]: temporary working memory,
#     [2]: output values.
RawProgramFunction = Callable[[ct.POINTER, ct.POINTER, ct.POINTER], None]


@dataclass
class RawProgram:
    """
    A raw program is returned by a circuit compiler to provide execution of
    the function defined by a compiled circuit.

    A `RawProgram` is a `Callable` with the signature:

    Fields:
        function: is a function of three ctypes arrays, returning nothing.
        dtype: the numpy data type of  the array values.
        number_of_vars: the number of input values (first function argument).
        number_of_tmps: the number of working memory values (second function argument).
        number_of_results: the number of result values (third function argument).
        var_indices: maps the index of inputs (from 0 to self.number_of_vars - 1) to the index
            of the corresponding circuit var.
    """

    function: RawProgramFunction
    dtype: DTypeNumeric
    number_of_vars: int
    number_of_tmps: int
    number_of_results: int
    var_indices: Sequence[int]

    def __call__(self, in_vars: NDArrayNumeric | Sequence[int | float]) -> NDArrayNumeric:
        """
        Call the raw program as a function from an array to an array.
        """
        array_vars: NDArrayNumeric
        if isinstance(vars, np.ndarray):
            array_vars = in_vars
        else:
            array_vars = np.array(in_vars, dtype=self.dtype)
        if array_vars.shape != (self.number_of_vars,):
            raise ValueError(f'input array incorrect shape: got {array_vars.shape} expected ({self.number_of_vars},)')
        if array_vars.dtype != self.dtype:
            raise ValueError(f'input array incorrect dtype: got {array_vars.dtype} expected {self.dtype}')

        array_tmps: NDArrayNumeric = np.zeros(self.number_of_tmps, dtype=self.dtype)
        array_outs: NDArrayNumeric = np.zeros(self.number_of_results, dtype=self.dtype)

        ptr_type = ct.POINTER(np.ctypeslib.as_ctypes_type(self.dtype))
        c_array_vars = array_vars.ctypes.data_as(ptr_type)
        c_array_tmps = array_tmps.ctypes.data_as(ptr_type)
        c_array_outs = array_outs.ctypes.data_as(ptr_type)

        self.function(c_array_vars, c_array_tmps, c_array_outs)
        return array_outs
