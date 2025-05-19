from __future__ import annotations

import ctypes as ct
from typing import Sequence

import numpy as np

from ck.program.raw_program import RawProgram, RawProgramFunction
from ck.utils.np_extras import DTypeNumeric, NDArrayNumeric


class ProgramBuffer:
    """
    A ProgramBuffer wraps a RawProgram with pre-allocated input, tmp, and out buffers.
    The buffers are numpy arrays.
    """

    def __init__(self, program: RawProgram):
        self._raw_program: RawProgram = program

        # Allocate the buffers
        self._array_vars = np.zeros(self.number_of_vars, dtype=self.dtype)
        self._array_tmps = np.zeros(self.number_of_tmps, dtype=self.dtype)
        self._array_outs = np.zeros(self.number_of_results, dtype=self.dtype)

        # Access the c-buffers
        ptr_type = ct.POINTER(np.ctypeslib.as_ctypes_type(self.dtype))
        self._c_array_vars = self._array_vars.ctypes.data_as(ptr_type)
        self._c_array_tmps = self._array_tmps.ctypes.data_as(ptr_type)
        self._c_array_outs = self._array_outs.ctypes.data_as(ptr_type)

        # Keep a direct reference to the internal callable.
        self._function: RawProgramFunction = program.function

    def clone(self) -> ProgramBuffer:
        """
        Take a copy of this program buffer, with the same raw program
        and same input and output values, but with new memory allocation
        for the buffers.
        """
        clone = ProgramBuffer(self._raw_program)
        clone[:] = self._array_vars
        clone.results[:] = self._array_outs
        return clone

    @property
    def raw_program(self) -> RawProgram:
        """
        What is the wrapped Program.
        """
        return self._raw_program

    @property
    def dtype(self) -> DTypeNumeric:
        """
        What is the numpy data type of values.
        This is the same as numpy and ctypes `dtype`.
        """
        return self._raw_program.dtype

    @property
    def number_of_vars(self) -> int:
        """
        How many input values are there to the function.
        Each input value relates to a circuit VarNode, as
        per method `var_indices`.

        Returns:
            the number of input values.
        """
        return self._raw_program.number_of_vars

    @property
    def number_of_tmps(self) -> int:
        """
        How many temporary values are there to the function.

        Returns:
            the number of temporary values.
        """
        return self._raw_program.number_of_tmps

    @property
    def number_of_results(self) -> int:
        """
        How many output values are there from the function.

        Returns:
            the number of output values.
        """
        return self._raw_program.number_of_results

    @property
    def var_indices(self) -> Sequence[int]:
        """
        Get the circuit `VarNode.index` for each function input.

        Returns:
            a list of the circuit VarNode indices, co-indexed
            with the function input values.
        """
        return self._raw_program.var_indices

    @property
    def vars(self) -> NDArrayNumeric:
        """
        Return the input variables as a numpy array.
        Writing to the returned array will write to the input slots of the program buffer.

        Warning:
            the array is backed by the program buffer memory, not a copy.
        """
        return self._array_vars

    @property
    def results(self) -> NDArrayNumeric:
        """
        Return the results as a numpy array.

        Warning:
            the array is backed by the program buffer memory, not a copy.
        """
        return self._array_outs

    def compute(self) -> NDArrayNumeric:
        """
        Compute and return the results, as per `self.results`.

        Warning:
            the array is backed by the program buffer memory, not a copy.
        """
        self._function(self._c_array_vars, self._c_array_tmps, self._c_array_outs)
        return self._array_outs

    def __setitem__(self, idx: int | slice, value: float) -> None:
        """
        Set the value of the indexed input variable.
        """
        self._array_vars[idx] = value

    def __getitem__(self, idx: int | slice) -> NDArrayNumeric:
        """
        Get the value of the indexed input variable.
        """
        return self._array_vars[idx]

    def __len__(self) -> int:
        """
        Number of input variables.
        """
        return len(self._array_vars)

    def __getstate__(self):
        """
        Support for pickle.
        """
        return {
            '_raw_program': self._raw_program,
            '_array_vars': self._array_vars,
            '_array_tmps': self._array_tmps,
            '_array_outs': self._array_outs,
        }

    def __setstate__(self, state):
        """
        Support for pickle.
        """
        self._raw_program = state['_raw_program']
        self._array_vars = state['_array_vars']
        self._array_tmps = state['_array_tmps']
        self._array_outs = state['_array_outs']

        # Access the c-buffers
        ptr_type = ct.POINTER(self.dtype)
        self._c_array_vars = self._array_vars.ctypes.data_as(ptr_type)
        self._c_array_tmps = self._array_tmps.ctypes.data_as(ptr_type)
        self._c_array_outs = self._array_outs.ctypes.data_as(ptr_type)

        # Keep a direct reference to the internal callable.
        self._function: RawProgramFunction = self._raw_program.function
