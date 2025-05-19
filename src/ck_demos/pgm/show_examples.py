from ck import example
from ck.pgm import PGM

INDENT: str = '    '
PRECISION: int = 3
MAX_STATE_DIGITS = 21


def main() -> None:
    """
    Show all standard PGM examples
    """
    for example_name, pgm_class in example.ALL_EXAMPLES.items():
        pgm: PGM = pgm_class()

        print(example_name)
        pgm.dump_synopsis(prefix=INDENT)
        print()

    print()
    print('Done.')


if __name__ == '__main__':
    main()
