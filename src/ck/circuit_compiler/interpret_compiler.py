from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Optional, Dict, List, Tuple, Callable

import numpy as np
import ctypes as ct

from ..circuit import Circuit, CircuitNode, VarNode, OpNode, ADD, MUL
from ..program.raw_program import RawProgram, RawProgramFunction
from ..utils.iter_extras import multiply, first
from ..utils.np_extras import NDArrayNumeric, DTypeNumeric
from .support.circuit_analyser import CircuitAnalysis, analyze_circuit
from .support.input_vars import InputVars, InferVars, infer_input_vars

# index to a value array
_VARS = 0
_CONSTS = 1
_TMPS = 2
_RESULT = 3


def compile_circuit(
        *result: CircuitNode,
        input_vars: InputVars = InferVars.ALL,
        circuit: Optional[Circuit] = None,
        dtype: DTypeNumeric = np.double,
) -> InterpreterRawProgram:
    """
    Make a RawProgram that interprets the given circuit.

    Args:
        *result: result nodes nominating the results of the returned program.
        input_vars: How to determine the input variables.
        circuit: optionally explicitly specify the Circuit.
        dtype: the numpy DType to use for the raw program.

    Returns:
        a raw program.

    Raises:
        ValueError: if the circuit is unknown, but it is needed.
        ValueError: if not all nodes are from the same circuit.
    """
    in_vars: Sequence[VarNode] = infer_input_vars(circuit, result, input_vars)
    analysis: CircuitAnalysis = analyze_circuit(in_vars, result)
    instructions: List[_Instruction]
    np_consts: NDArrayNumeric
    instructions, np_consts = _make_instructions(analysis, dtype)

    return InterpreterRawProgram(
        in_vars=in_vars,
        result=result,
        op_nodes=analysis.op_nodes,
        dtype=dtype,
        instructions=instructions,
        np_consts=np_consts,
    )


class InterpreterRawProgram(RawProgram):
    def __init__(
            self,
            in_vars: Sequence[VarNode],
            result: Sequence[CircuitNode],
            op_nodes: Sequence[OpNode],
            dtype: DTypeNumeric,
            instructions: List[_Instruction],
            np_consts: NDArrayNumeric,
    ):
        self.instructions = instructions
        self.np_consts = np_consts

        function = _make_function(
            instructions=instructions,
            np_consts=np_consts,
        )

        super().__init__(
            function=function,
            dtype=dtype,
            number_of_vars=len(in_vars),
            number_of_tmps=len(op_nodes),
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
            'instructions': self.instructions,
            'np_consts': self.np_consts,
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
        self.instructions = state['instructions']
        self.np_consts = state['np_consts']

        self.function = _make_function(
            instructions=self.instructions,
            np_consts=self.np_consts,
        )


def _make_instructions(
        analysis: CircuitAnalysis,
        dtype: DTypeNumeric,
) -> Tuple[Sequence[_Instruction], NDArrayNumeric]:

    # Store const values in a numpy array
    node_to_const_idx: Dict[int, int] = {
        id(node): i
        for i, node in enumerate(analysis.const_nodes)
    }
    np_consts: NDArrayNumeric = np.zeros(len(node_to_const_idx), dtype=dtype)
    for i, node in enumerate(analysis.const_nodes):
        np_consts[i] = node.value

    # Where to get input values for each possible node.
    node_to_element: Dict[int, _ElementID] = {}
    # const nodes
    for node_id, const_idx in node_to_const_idx.items():
        node_to_element[node_id] = _ElementID(_CONSTS, const_idx)
    # var nodes
    var_node: VarNode
    for i, var_node in enumerate(analysis.var_nodes):
        if var_node.is_const():
            node_to_element[id(var_node)] = node_to_element[id(var_node.const)]
        else:
            node_to_element[id(var_node)] = _ElementID(_VARS, i)
    # op nodes
    for node_id, tmp_index in analysis.op_to_tmp.items():
        node_to_element[node_id] = _ElementID(_TMPS, tmp_index)
    for node_id, tmp_index in analysis.op_to_result.items():
        node_to_element[node_id] = _ElementID(_RESULT, tmp_index)

    # Build instructions
    instructions: List[_Instruction] = []

    op_node: OpNode
    for op_node in analysis.op_nodes:
        dest: _ElementID = node_to_element[id(op_node)]
        args: List[_ElementID] = [
            node_to_element[id(arg)]
            for arg in op_node.args
        ]
        if op_node.symbol == MUL:
            operation = multiply
        elif op_node.symbol == ADD:
            operation = sum
        else:
            assert False, 'symbol not understood'

        instructions.append(_Instruction(operation, args, dest))

    # Add any copy operations, i.e., result nodes that are not op nodes
    for i, node in enumerate(analysis.result_nodes):
        if not isinstance(node, OpNode):
            source: _ElementID = node_to_element[id(node)]
            instructions.append(_Instruction(first, [source], _ElementID(_RESULT, i)))

    return instructions, np_consts


def _make_function(
        instructions: List[_Instruction],
        np_consts: NDArrayNumeric,
) -> RawProgramFunction:
    """
    Make a RawProgram function that executes the given instructions.
    """

    # RawProgramFunction = Callable[[ct.POINTER, ct.POINTER, ct.POINTER], None]
    def raw_program_function(vars_in: ct.POINTER, tmps: ct.POINTER, result_out: ct.POINTER) -> None:
        nonlocal np_consts
        nonlocal instructions

        arrays: List[ct.POINTER] = [None, None, None, None]
        arrays[_VARS] = vars_in
        arrays[_TMPS] = tmps
        arrays[_RESULT] = result_out
        arrays[_CONSTS] = np_consts

        def get_value(_element: _ElementID):
            return arrays[_element.array][_element.index]

        instruction: _Instruction
        element: _ElementID
        for instruction in instructions:
            value = instruction.operation(get_value(element) for element in instruction.args)
            dest: _ElementID = instruction.dest
            arrays[dest.array][dest.index] = value

    return raw_program_function


@dataclass
class _ElementID:
    array: int  # VARS, TMPS, CONSTS, RESULT
    index: int  # index into the array


@dataclass
class _Instruction:
    operation: Callable
    args: Sequence[_ElementID]
    dest: _ElementID
