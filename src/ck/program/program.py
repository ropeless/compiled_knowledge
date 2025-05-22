"""
For more documentation on this module, refer to the Jupyter notebook docs/6_circuits_and_programs.ipynb.
"""
from typing import Callable, Sequence

import numpy as np

from ck.program.raw_program import RawProgram
from ck.utils.np_extras import DTypeNumeric, NDArrayNumeric


class Program:
    """
    A program represents an arithmetic a function from input values to output values.

    Internally a `Program` wraps a `RawProgram` which is the object returned by a circuit compiler.

    Every `Program` has a numpy `dtype` which defines the numeric data type for input and output values.
    Typically, the `dtype` of a program is a C style double.
    """

    def __init__(self, raw_program: RawProgram):
        self._raw_program = raw_program
        self.__call = self.__get_call_method()

    @property
    def dtype(self) -> DTypeNumeric:
        """
        What is the C data type of values.
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

    def __call__(self, *args: float | int) -> NDArrayNumeric | int | float:
        """
        Call the compiled program.

        Returns:
            either a single value or a numpy array, depending
            on the construction arguments.

        Raises:
            ValueError: if the given number of argument != self.number_of_vars.
        """
        # dynamically defined at construction time
        return self.__call(*args)

    def __call_single(self, *args: float) -> int | float:
        """
        Returns a single result of type self._dtype.value.dtype
        """
        return self._raw_program(args).item()

    def __call_empty(self, *args: float | int) -> NDArrayNumeric:
        """
        Returns a numpy array result of dtype self.dtype
        """
        if len(args) != self.number_of_vars:
            raise ValueError(f'incorrect number of arguments: expected {self.number_of_vars}, got {len(args)}')
        np_out = np.zeros(0, dtype=self.dtype)
        return np_out

    def __call_multi(self, *args: float | int) -> NDArrayNumeric:
        """
        Returns a numpy array result of dtype self.dtype
        """
        return self._raw_program(args)

    def __get_call_method(self) -> Callable:
        """
        Choose a call method based on self._number_of_results
        """
        match self.number_of_results:
            case 0:
                return self.__call_empty
            case 1:
                return self.__call_single
            case _:
                return self.__call_multi

    def __getstate__(self):
        """
        Support for pickle.
        """
        return {
            '_raw_program': self._raw_program,
        }

    def __setstate__(self, state):
        """
        Support for pickle.
        """
        self._raw_program = state['_raw_program']
        self.__call = self.__get_call_method()
