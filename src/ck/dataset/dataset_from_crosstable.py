from typing import Sequence

import numpy as np

from ck.dataset import HardDataset, SoftDataset
from ck.dataset.cross_table import CrossTable, cross_table_from_soft_dataset
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


def expand_soft_dataset(soft_dataset: SoftDataset) -> HardDataset:
    """
    Construct a hard dataset with the same data semantics as the given soft dataset
    by expanding soft evidence.

    Any state weights in `soft_dataset` that represents uncertainty over states
    of a random variable will be converted to an equivalent set of weighted hard
    instances. This means that the returned dataset may have a number of instances
    different to that of the given soft dataset.

    The ordering of instances in the returned dataset is not guaranteed.

    This method works by constructing a cross-table from the given soft dataset,
    then converting the crosstable to a hard dataset using `dataset_from_cross_table`.
    This implies that the result will have no duplicated instances and no
    instances with weight zero.
    """
    crosstab: CrossTable = cross_table_from_soft_dataset(soft_dataset)
    return dataset_from_cross_table(crosstab)
