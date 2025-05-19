"""
This demo shows how to dump a Circuit.
"""
from ck.circuit import Circuit


def main() -> None:
    cct = Circuit()
    x0 = cct.new_var()  # this var will have index 0
    x1 = cct.new_var()  # this var will have index 1
    c123 = cct.const(123)
    m = cct.mul(x0, x1)
    _ = cct.add(c123, m)

    cct.dump()

    # with open('demo_cct_dump.txt', 'w') as out:
    #     circuit.dump(out)


if __name__ == '__main__':
    main()
