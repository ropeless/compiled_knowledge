"""
This demo script can be used to see how well samplers perform.
"""
import random

from ck.pgm import PGM
from ck.utils.random_extras import random_permutation
from ck_demos.utils.sample_model import sample_model, STANDARD_SAMPLERS, BURN_IN

# Parameters for random PGM
num_of_rvs = 5
states_per_rv = 3
proportion_zeros = 0.1

# Experiment parameters
num_of_trials = 3
limit_conditioning = None
num_of_samples = 100 * (states_per_rv ** num_of_rvs)
seed = None

# Formatting options
show_each_analysis = False


def main():
    # Manage randomness, if requested
    if seed is not None:
        random.seed(seed)

    # Create a random PGM
    pgm = PGM()
    rvs = [pgm.new_rv(f'x_{i}', states_per_rv) for i in range(num_of_rvs)]

    # Add unary factors to all but the first rv (uniform prior).
    for rv in rvs[1:]:
        pgm.new_factor(rv).set_dense().set_uniform()
    # Make the first rv the child of all others (random probabilities).
    f = pgm.new_factor(*rvs).set_dense().set_stream(random.random)

    # Force some zero probabilities.
    num_to_zero = int(f.number_of_parameters * proportion_zeros + 0.5)
    if num_to_zero > 0:
        to_zero = random_permutation(f.number_of_parameters)[:num_to_zero]
        for param_idx in to_zero:
            f.set_param_value(param_idx, 0)

    # Make the PGM a Bayesian network.
    f.normalise_cpt()

    samplers_str = ', '.join([name for name in STANDARD_SAMPLERS.keys()])
    print(f'Samplers:                   {samplers_str}')
    print(f'Number of trials:           {num_of_trials}')
    print(f'Number of samples:          {num_of_samples}')
    print(f'States per RV:              {states_per_rv}')
    print(f'Burn in (where used):       {BURN_IN}')
    print(f'Random seed:                {seed}')

    sample_model(
        pgm,
        STANDARD_SAMPLERS,
        num_of_trials,
        num_of_samples,
        limit_conditioning=limit_conditioning,
        show_each_analysis=show_each_analysis
    )

    print('Done.')


if __name__ == '__main__':
    main()
