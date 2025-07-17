from ck.dataset import HardDataset, SoftDataset
from ck.dataset.dataset_builder import DatasetBuilder, soft_dataset_from_builder, hard_dataset_from_builder
from ck.pgm import PGM


def main() -> None:
    pgm = PGM()
    x = pgm.new_rv('x', (True, False))
    y = pgm.new_rv('y', ('yes', 'no', 'maybe'))

    builder = DatasetBuilder([x, y])
    builder.append()
    builder.append(1, 2).weight = 3
    builder.append(None, [0.7, 0.1, 0.2])
    builder.append().set_states(True, 'maybe')

    print('DatasetBuilder dump')
    builder.dump()
    print()

    print('DatasetBuilder dump, showing states and custom missing values')
    builder.dump(as_states=True, missing='?')
    print()

    print('HardDataset dump')
    dataset: HardDataset = hard_dataset_from_builder(builder, missing=99)
    dataset.dump()
    print()

    print('SoftDataset dump')
    dataset: SoftDataset = soft_dataset_from_builder(builder)
    dataset.dump()
    print()


if __name__ == '__main__':
    main()
