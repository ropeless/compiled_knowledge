import random
from typing import Optional, Dict, Callable, List

import numpy as np

from ck.pgm import rv_instances, PGM, RandomVariable, Indicator
from ck.pgm_compiler import factor_elimination
from ck.pgm_circuit.marginals_program import MarginalsProgram
from ck.pgm_circuit import PGMCircuit
from ck.pgm_circuit.wmc_program import WMCProgram
from ck.sampling.forward_sampler import ForwardSampler
from ck.sampling.sampler import Sampler
from ck.utils.random_extras import random_permute
from ck_demos.utils.stop_watch import StopWatch

SamplerFactory = Callable[[PGM, WMCProgram, MarginalsProgram, List[RandomVariable], List[Indicator]], Sampler]

BURN_IN: int = 1000  # Burn in for standard samplers, where needed. Not all samplers use burn in.

# Standard Samplers (by name)
STANDARD_SAMPLERS: Dict[str, SamplerFactory] = {
    'Direct-wmc': (
        lambda pgm, wmc, mar, sample_rvs, condition:
        wmc.sample_direct(rvs=sample_rvs, condition=condition)
    ),
    'Direct-mar': (
        lambda pgm, wmc, mar, sample_rvs, condition:
        mar.sample_direct(rvs=sample_rvs, condition=condition)
    ),
    'Rejection': (
        lambda pgm, wmc, mar, sample_rvs, condition:
        wmc.sample_rejection(rvs=sample_rvs, condition=condition)
    ),
    'Gibbs': (
        lambda pgm, wmc, mar, sample_rvs, condition:
        wmc.sample_gibbs(burn_in=BURN_IN, rvs=sample_rvs, condition=condition)
    ),
    'Metropolis': (
        lambda pgm, wmc, mar, sample_rvs, condition:
        wmc.sample_metropolis(burn_in=BURN_IN, rvs=sample_rvs, condition=condition)
    ),
    'Forward': (
        lambda pgm, wmc, mar, sample_rvs, condition:
        ForwardSampler(pgm, sample_rvs, condition, check_is_bayesian_network=True)
    ),
    'Uniform': (
        lambda pgm, wmc, mar, sample_rvs, condition:
        wmc.sample_uniform(rvs=sample_rvs, condition=condition)
    ),
}


def sample_model(
        pgm: PGM,
        samplers: Dict[str, SamplerFactory],
        num_of_trials: int,
        num_of_samples: int,
        limit_conditioning: Optional[int] = None,
        show_each_analysis: bool = True,
        line: str = '-' * 80,
):
    """
    Evaluate the given samplers on the given PGM.

    Results are printed to standard out.

    Args:
        pgm: is the model to sample.
        samplers: is a dict from sampler name to factory method. The
            factor method type is (pgm, wmc, mar, sample_rvs, condition) -> Sampler.
        num_of_trials: how many trials to perform.
        num_of_samples: how many num_of_samples to draw from each sampler, for each trial.
        limit_conditioning: maximum number of indicators to use when determining
            conditioning for a trial, or None then pgm.number_of_random_variables is used.
        show_each_analysis: if True, then extra details is printed.
        line: is the 'line' string to use to delimit trials.
    """
    print(f'Model:                      {pgm.name}')
    print(f'Number of random variables: {pgm.number_of_rvs}')
    print(f'Number of indicators:       {pgm.number_of_indicators}')
    print(f'States space:               {pgm.number_of_states:,}')

    # compile
    pgm_cct: PGMCircuit = factor_elimination.compile_pgm(pgm)
    wmc = WMCProgram(pgm_cct)
    mar = MarginalsProgram(pgm_cct)

    rvs = pgm.rvs
    num_of_rvs = len(rvs)
    sampler_names = list(samplers.keys())
    overall_max_difference = {name: 0 for name in sampler_names}
    overall_sum_difference = {name: 0 for name in sampler_names}
    overall_time = {name: 0 for name in sampler_names}
    errors = {name: [] for name in sampler_names}

    name_pad = max(
        max(len(name) for name in sampler_names) + 1,
        max(len(rv.name) for rv in rvs) + 1
    )

    for trial in range(1, 1 + num_of_trials):
        print(line)

        # what random variables to sample
        num_rvs_to_sample = random.randint(1, num_of_rvs)
        sample_rvs = list(rvs)
        random_permute(sample_rvs)
        del sample_rvs[num_rvs_to_sample:]
        sample_rvs.sort(key=(lambda rv: rv.idx))
        rvs_str = ', '.join([str(rv) for rv in sample_rvs])

        # what conditions
        if limit_conditioning is None:
            limit_conditioning = pgm.number_of_rvs
        if limit_conditioning == 0:
            condition = ()
            condition_str = ''
        else:
            while True:
                num_indicators_to_condition = random.randint(0, limit_conditioning)
                rand_rvs = list(rvs)
                random_permute(rand_rvs)
                condition = []
                while len(condition) < num_indicators_to_condition and len(rand_rvs) > 0:
                    rv = rand_rvs.pop()
                    max_rv_indicators_to_condition = min(len(rv) - 1, num_indicators_to_condition - len(condition))
                    assert max_rv_indicators_to_condition >= 1, 'assumption check'
                    num_rv_indicators_to_condition = random.randint(1, max_rv_indicators_to_condition)
                    indicators = list(rv)
                    random_permute(indicators)
                    condition += sorted(indicators[:num_rv_indicators_to_condition])

                if len(condition) == 0:
                    condition_str = ''
                    break

                condition_str = ' | ' + pgm.condition_str(*condition)

                # only accept the condition if the Pr(condition) > 0
                if wmc.probability(*condition) > 0:
                    break
                print(f'Note: discarded impossible condition{condition_str}')

        # show the trial parameters
        print(f'trial {trial} of {num_of_trials}: {rvs_str}{condition_str}')

        # create state indexes for printing
        state_to_index = {}
        all_states = []
        for i, state in enumerate(rv_instances(*sample_rvs)):
            state = tuple(state)
            all_states.append(state)
            state_to_index[state] = i

        # print detailed results - header
        for i, rv in enumerate(sample_rvs):
            print(str(rv).ljust(name_pad), end='')
            print(' '.join([f'{str(state[i]).ljust(7)}' for state in all_states]))

        # pgm_stats
        print('PGM'.ljust(name_pad), end='')
        pgm_stats = np.array(wmc.marginal_distribution(*sample_rvs, condition=condition))
        print(' '.join([f'{p:.5f}' for p in pgm_stats]))

        for sampler_name in sampler_names:
            print(sampler_name.ljust(name_pad), end='')

            # sample_stats
            try:
                sample_stats = np.zeros(len(all_states))
                sampler = samplers[sampler_name](pgm, wmc, mar, sample_rvs, condition)
                stop_watch = StopWatch()
                for state in sampler.take(num_of_samples):
                    i = state_to_index[tuple(state)]
                    sample_stats[i] += 1
                stop_watch.stop()
                sample_stats /= np.sum(sample_stats)
            except (ValueError, RuntimeError, AssertionError) as err:
                errors[sampler_name].append(repr(err))
                print(repr(err))
                continue

            # print detailed results - for this sampler
            print(' '.join([f'{p:.5f}' for p in sample_stats]))

            # analyse
            max_difference = 0
            sum_difference = 0
            for pgm_stat, sample_stat in zip(pgm_stats, sample_stats):
                diff = abs(pgm_stat - sample_stat)
                max_difference = max(max_difference, diff)
                sum_difference += diff
            overall_max_difference[sampler_name] = max(overall_max_difference[sampler_name], max_difference)
            overall_sum_difference[sampler_name] = max(overall_sum_difference[sampler_name], sum_difference)
            overall_time[sampler_name] += stop_watch.seconds()

            if show_each_analysis:
                print(
                    ' ' * name_pad +
                    f'max_difference = {max_difference}, '
                    f'sum_difference = {sum_difference}, '
                    f'time = {stop_watch.seconds()}'
                )

    print(line)
    sep: str = ', '
    print(' ' * name_pad + sep.join(['overall_max_difference', 'overall_sum_difference', 'overall_time', 'errors']))
    for sampler_name in sampler_names:
        print(
            f'{sampler_name.ljust(name_pad)}'
            f'{overall_max_difference[sampler_name]}{sep}'
            f'{overall_sum_difference[sampler_name]}{sep}'
            f'{overall_time[sampler_name]}{sep}'
            f'{len(errors[sampler_name])}'
        )
    print()
