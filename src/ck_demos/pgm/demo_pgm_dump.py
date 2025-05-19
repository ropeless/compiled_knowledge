from ck import example


def main() -> None:
    """
    Demonstrate dumping a PGM
    """
    pgm = example.Alarm()

    print('Dumping PGM')
    pgm.dump()

    print()
    print('Done.')


if __name__ == '__main__':
    main()
