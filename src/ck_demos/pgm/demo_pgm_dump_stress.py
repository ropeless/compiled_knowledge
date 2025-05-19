"""
Demonstrate dumping a PGM - stressing the method
"""
from ck import example


def main() -> None:
    pgm = example.Stress()

    print('Dumping PGM')
    pgm.dump(show_function_values=True)

    print()
    print('Done.')


if __name__ == '__main__':
    main()
