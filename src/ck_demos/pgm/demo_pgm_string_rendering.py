from ck.pgm import PGM


def main() -> None:
    pgm = PGM()

    a = pgm.new_rv('A', ('x', 'y', 'z'))
    b = pgm.new_rv('B', (3, 5))

    print(pgm.indicator_str(a[0], b[1], a[2]))
    print(pgm.condition_str(a[0], b[1], a[2]))


if __name__ == '__main__':
    main()
