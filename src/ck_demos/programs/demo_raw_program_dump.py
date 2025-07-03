from ck.circuit import Circuit
from ck.circuit_compiler import NamedCircuitCompiler


def main() -> None:
    cct = Circuit()
    a, b, c, d = cct.new_vars(4)
    top = a * b + c * d + 56.23

    for compiler in NamedCircuitCompiler:
        raw_program = compiler(top)
        raw_program.dump()
        print()


if __name__ == '__main__':
    main()
