from ck.circuit_compiler.interpret_compiler import compile_circuit
from ck.circuit import Circuit
from ck.program import Program
from ck.program.raw_program import RawProgram


def main() -> None:
    circuit = Circuit()
    input_vars = circuit.new_vars(4)

    raw_program: RawProgram = compile_circuit(input_vars=input_vars)
    prog = Program(raw_program)

    result = prog(4, 5, 6, 7)
    print('result =', result)


if __name__ == '__main__':
    main()
