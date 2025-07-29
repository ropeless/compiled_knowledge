from ck import example
from ck.dataset.sampled_dataset import dataset_from_sampler
from ck.pgm import PGM
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.pgm_compiler import DEFAULT_PGM_COMPILER
from ck.sampling.sampler import Sampler


def main() -> None:
    pgm: PGM = example.Student()
    sampler: Sampler = WMCProgram(DEFAULT_PGM_COMPILER(pgm)).sample_direct()
    dataset = dataset_from_sampler(sampler, 10)

    dataset.dump()


if __name__ == '__main__':
    main()
