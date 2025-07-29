from ck.dataset.dataset_from_csv import hard_dataset_from_csv
from ck.learning.train_generative_bn import train_generative_bn
from ck.pgm import PGM


def main() -> None:
    pgm = PGM('Student')

    difficult = pgm.new_rv('difficult', ['y', 'n'])
    intelligent = pgm.new_rv('intelligent', ['y', 'n'])
    grade = pgm.new_rv('grade', ['low', 'medium', 'high'])
    award = pgm.new_rv('award', ['y', 'n'])
    letter = pgm.new_rv('letter', ['y', 'n'])

    pgm.new_factor(difficult)
    pgm.new_factor(intelligent)
    pgm.new_factor(grade, intelligent, difficult)
    pgm.new_factor(award, intelligent)
    pgm.new_factor(letter, grade)

    rvs = (difficult, intelligent, grade, award, letter)
    csv = """
    0,1,2,0,1
    1,1,2,0,1
    1,1,2,0,1
    0,0,2,0,0
    0,1,1,1,0
    1,1,1,1,1
    1,1,0,0,0
    1,1,0,0,1
    1,0,0,0,0
    """

    dataset = hard_dataset_from_csv(rvs, csv.splitlines())

    # Learn parameters values for `pgm` using the training data `dataset`.
    # This updates the PGMs potential functions.
    train_generative_bn(pgm, dataset)

    show_pgm_factors(pgm)

    print('Done.')


def show_pgm_factors(pgm: PGM) -> None:
    for factor in pgm.factors:
        potential_function = factor.function
        print(f'Factor: {factor} {type(potential_function)}')
        for instance, _, param_value in potential_function.keys_with_param:
            print(f'Factor{instance} = {param_value}')
        print()


if __name__ == '__main__':
    main()
