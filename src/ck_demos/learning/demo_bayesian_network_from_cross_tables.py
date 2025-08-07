from typing import List, Set

from ck import example
from ck.dataset import HardDataset
from ck.dataset.cross_table import CrossTable, cross_table_from_hard_dataset
from ck.dataset.sampled_dataset import dataset_from_sampler
from ck.learning.model_from_cross_tables import model_from_cross_tables
from ck.pgm import PGM, RandomVariable
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import DEFAULT_PGM_COMPILER
from ck.probability import divergence

EXCLUDE_UNNECESSARY_CROSS_TABLES = True


def main() -> None:
    # Create a dataset based on model which is an example PGM
    number_of_samples: int = 10000  # How many instances to make for the model dataset
    model: PGM = example.Student()
    model_dataset: HardDataset = dataset_from_sampler(
        WMCProgram(DEFAULT_PGM_COMPILER(model)).sample_direct(),
        number_of_samples,
    )

    # Clone the model, without factors, and transport the dataset to the new PGM
    pgm = PGM()
    dataset = HardDataset(weights=model_dataset.weights)
    for model_rv in model.rvs:
        rv = pgm.new_rv(model_rv.name, model_rv.states)
        dataset.add_rv_from_state_idxs(rv, model_dataset.state_idxs(model_rv))

    # What model rvs have a child
    model_rvs_with_children: Set[RandomVariable] = set()
    for model_factor in model.factors:
        for parent_rv in model_factor.rvs[1:]:
            model_rvs_with_children.add(parent_rv)

    # Construct cross-tables from the dataset
    cross_tables: List[CrossTable] = []
    for model_factor in model.factors:
        if (
                EXCLUDE_UNNECESSARY_CROSS_TABLES
                and len(model_factor.rvs) == 1
                and model_factor.rvs[0] in model_rvs_with_children
        ):
            # The factor relates to a single random variable (has
            # no parents) but it does have children.
            # No need to include a cross-table as it is inferable from
            # cross-tables of its children.
            continue

        rvs = tuple(pgm.rvs[model_rv.idx] for model_rv in model_factor.rvs)
        cross_tables.append(cross_table_from_hard_dataset(dataset, rvs))
        print('cross-table:', *rvs)

    # Train the PGM
    model_from_cross_tables(pgm, cross_tables)

    # Show results
    print()
    pgm.dump(show_function_values=True)
    print()
    model_space = WMCProgram(DEFAULT_PGM_COMPILER(model))
    pgm_space = WMCProgram(DEFAULT_PGM_COMPILER(pgm))
    print('HI', divergence.hi(model_space, pgm_space))
    print('KL', divergence.kl(model_space, pgm_space))


if __name__ == '__main__':
    main()
