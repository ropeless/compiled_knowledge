from typing import Sequence

import numpy as np

from ck.dataset import HardDataset
from ck.dataset.cross_table import CrossTable
from ck.pgm import RandomVariable
from ck.utils.np_extras import dtype_for_number_of_states


def dataset_from_cross_table(cross_table: CrossTable) -> HardDataset:
    """
    Construct a HardDataset from the given cross-table.

    Args:
        cross_table: A cross-table represented as a dictionary.

    Returns:
        A dataset where instances and instance weights are those of the
        given cross-table.

    Ensures:
        `result.total_weight() == dataset.total_weight()`.
        Zero weighted instances are not counted.
    """
    rvs: Sequence[RandomVariable] = cross_table.rvs

    # Unzip the cross-table dictionary
    rvs_series = [[] for _ in range(len(rvs))]
    weights = []
    for instance, weight in cross_table.items():
        for series, state in zip(rvs_series, instance):
            series.append(state)
        weights.append(weight)

    # Put the hard dataset together
    return HardDataset(
        data=(
            (rv, np.array(series, dtype=dtype_for_number_of_states(len(rv))))
            for rv, series in zip(rvs, rvs_series)
        ),
        weights=np.array(weights, dtype=np.float64),
    )


