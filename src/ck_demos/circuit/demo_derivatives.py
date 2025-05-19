"""
This is a script for exploring partial derivatives.
"""
from ck.circuit_compiler.llvm_compiler import compile_circuit
from ck.circuit import Circuit
from ck.program import Program


def main() -> None:
    cct = Circuit()
    x = cct.new_vars(2)  # indicators x[0] and x[1]
    y = cct.new_vars(2)  # indicators y[0] and y[1]
    f = cct.new_vars(2)  # factor between x and y
    q = cct.add(
        cct.mul(x[0], y[0], f[0]),
        cct.mul(x[0], y[1], f[1]),
        cct.mul(x[1], y[0], f[1]),
        cct.mul(x[1], y[1], f[0]),
    )

    derivatives = cct.partial_derivatives(q, f)  # returns a list: [f0', f1']

    # prog is a function from [x0, x1, y0, y1, f0, f1] to [f0', f1', q]
    prog = Program(compile_circuit(*derivatives + [q]))

    # Make a dataset.
    dataset = [
        # x    y     f
        [1, 0, 1, 0, 0.4, 0.6],
        [0, 1, 1, 0, 0.4, 0.6],
        [1, 0, 0, 1, 0.4, 0.6],
        [0, 1, 0, 1, 0.4, 0.6],
        [1, 1, 1, 1, 0.4, 0.6],
    ]

    for instance in dataset:
        result = prog(*instance)
        print(instance, result[:-1], result[-1])
    print()


if __name__ == '__main__':
    main()
