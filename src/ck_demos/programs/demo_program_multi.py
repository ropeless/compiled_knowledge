from ck.circuit_compiler.interpret_compiler import compile_circuit
from ck.circuit import Circuit
from ck.program import Program
from ck.program.raw_program import RawProgram


def main() -> None:
    cct = Circuit()
    x = cct.new_vars(4)
    y = cct.add(x[0], x[1])
    z = cct.add(x[2], x[3])
    top1 = cct.add(1, cct.mul(y, -12))
    top2 = cct.add(1, cct.mul(z, 12))

    raw_program: RawProgram = compile_circuit(top1, top2)
    prog = Program(raw_program)

    result = prog(4, 5, 6, 7)
    print('expect = [-107, 157]')
    print('result =', result)


if __name__ == '__main__':
    main()
