from typing import Optional, Sequence, Tuple

import numpy as np

from ck.circuit import CircuitNode, Circuit, VarNode
from ck.circuit_compiler.support.input_vars import InferVars, InputVars, infer_input_vars
from ck.program.raw_program import RawProgram, RawProgramFunction
from ck.utils.np_extras import DTypeNumeric
from . import _compiler
from ..support.circuit_analyser import CircuitAnalysis, analyze_circuit


def compile_circuit(
        *result: CircuitNode,
        input_vars: InputVars = InferVars.ALL,
        circuit: Optional[Circuit] = None,
        dtype: DTypeNumeric = np.double,
) -> RawProgram:
    """
    Make a RawProgram that interprets the given circuit using a Cython virtual cpu.

    Args:
        *result: if it is a single node, then the resulting function returns a single value,
            if it is a sequence of nodes, then the resulting function returns a numpy array of values.
        input_vars: how to determine the input variables.
        circuit: optionally explicitly specify the Circuit
        dtype: the numpy DType to use for the raw program.

    Returns:
        a raw program.

    Raises:
        ValueError: if the circuit is unknown, but it is needed.
        ValueError: if not all nodes are from the same circuit.
        ValueError: if the given dtype is not supported.
    """
    if dtype not in _compiler.DTYPE_TO_CVM_TYPE.keys():
        raise ValueError(f'dtype not supported: {dtype!r}')

    in_vars: Sequence[VarNode] = infer_input_vars(circuit, result, input_vars)
    return CythonRawProgram(in_vars, result, dtype)


class CythonRawProgram(RawProgram):
    def __init__(
            self,
            in_vars: Sequence[VarNode],
            result: Sequence[CircuitNode],
            dtype: DTypeNumeric,
    ):
        self.in_vars = in_vars
        self.result = result

        function, number_of_tmps = _make_function(
            var_nodes=in_vars,
            result_nodes=result,
            dtype=dtype,
        )

        super().__init__(
            function=function,
            dtype=dtype,
            number_of_vars=len(in_vars),
            number_of_tmps=number_of_tmps,
            number_of_results=len(result),
            var_indices=tuple(var.idx for var in in_vars),
        )

    def __getstate__(self):
        """
        Support for pickle.
        """
        return {
            'dtype': self.dtype,
            'number_of_vars': self.number_of_vars,
            'number_of_tmps': self.number_of_tmps,
            'number_of_results': self.number_of_results,
            'var_indices': self.var_indices,
            #
            'in_vars': self.in_vars,
            'result': self.result,
        }

    def __setstate__(self, state):
        """
        Support for pickle.
        """
        self.dtype = state['dtype']
        self.number_of_vars = state['number_of_vars']
        self.number_of_tmps = state['number_of_tmps']
        self.number_of_results = state['number_of_results']
        self.var_indices = state['var_indices']
        #
        self.in_vars = state['in_vars']
        self.result = state['result']

        self.function, _ = _make_function(
            var_nodes=self.in_vars,
            result_nodes=self.result,
            dtype=self.dtype,
        )


def _make_function(
        var_nodes: Sequence[VarNode],
        result_nodes: Sequence[CircuitNode],
        dtype: DTypeNumeric,
) -> Tuple[RawProgramFunction, int]:
    """
    Make a RawProgram function that interprets the circuit.

    Args:
        var_nodes: The chosen input variable nodes of the circuit.
        result_nodes: The chosen output result nodes of the circuit.
        dtype: a numpy data type that must be a key in the dictionary, DTYPE_TO_CVM_TYPE.

    Returns:
        (function, number_of_tmps)
    """
    analysis: CircuitAnalysis = analyze_circuit(var_nodes, result_nodes)
    return _compiler.make_function(analysis, dtype)
