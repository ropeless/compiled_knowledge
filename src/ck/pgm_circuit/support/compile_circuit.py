from typing import Optional, Sequence

from ck.circuit import CircuitNode, TmpConst, Circuit
from ck.circuit_compiler import CircuitCompiler
from ck.circuit_compiler import DEFAULT_CIRCUIT_COMPILER
from ck.pgm_circuit import PGMCircuit
from ck.program import RawProgram


def compile_results(
        pgm_circuit: PGMCircuit,
        results: Sequence[CircuitNode],
        const_parameters: bool,
        compiler: CircuitCompiler = DEFAULT_CIRCUIT_COMPILER,
) -> RawProgram:
    """
    Compile a circuit to a raw program that calculates the given result.

    Raises:
        ValueError: if not all nodes are from the same circuit.

    Args:
        pgm_circuit: The circuit (and PGM) that will be compiled to a program.
        results: the result circuit nodes for the returned program.
        const_parameters: if True then any circuit variable representing a parameter value will
            be made 'const' in the resulting program.
        compiler: function from circuit nodes to raw program.

    Returns:
        a compiled RawProgram.
    """
    circuit: Circuit = pgm_circuit.circuit_top.circuit
    if const_parameters:
        parameter_values = pgm_circuit.parameter_values
        number_of_indicators = pgm_circuit.number_of_indicators
        with TmpConst(circuit) as tmp:
            for slot, value in enumerate(parameter_values, start=number_of_indicators):
                tmp.set_const(slot, value)
            raw_program: RawProgram = compiler(*results, circuit=circuit)
    else:
        raw_program: RawProgram = compiler(*results, circuit=circuit)

    return raw_program


def compile_param_derivatives(
        pgm_circuit: PGMCircuit,
        self_multiply: bool = False,
        params_value: Optional[float | int] = 1,
        compiler: CircuitCompiler = DEFAULT_CIRCUIT_COMPILER,
) -> RawProgram:
    """
    Compile the circuit to a program for computing the partial derivatives of the parameters.
    partial derivatives are co-indexed with pgm_circuit.parameter_values.

    Typically, this will grow the circuit by the addition of circuit nodes to compute the derivatives.

    Args:
        pgm_circuit: The circuit (and PGM) that will be compiled to a program.
        self_multiply: if true then each partial derivative df/dx will be multiplied by x.
        params_value: if not None, then circuit vars representing parameters will be temporarily
            set to this value for compiling the program. Default is 1.
        compiler: function from circuit nodes to raw program.
    """
    top: CircuitNode = pgm_circuit.circuit_top
    circuit: Circuit = top.circuit

    start_idx = pgm_circuit.number_of_indicators
    end_idx = start_idx + pgm_circuit.number_of_parameters
    param_vars = circuit.vars[start_idx:end_idx]
    derivatives = circuit.partial_derivatives(top, param_vars, self_multiply=self_multiply)

    if params_value is not None:
        with TmpConst(circuit) as tmp:
            tmp.set_const(param_vars, params_value)
            raw_program: RawProgram = compiler(*derivatives, circuit=circuit)
    else:
        raw_program: RawProgram = compiler(*derivatives, circuit=circuit)

    return raw_program
