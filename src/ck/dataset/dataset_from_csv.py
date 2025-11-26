from typing import Iterable, List, Sequence, Optional

from ck.dataset import HardDataset
from ck.pgm import RandomVariable


def hard_dataset_from_csv(
        rvs: Iterable[RandomVariable],
        lines: Iterable[str],
        *,
        weights: Optional[int | str] = None,
        sep: Optional[str] = ',',
        comment: str = '#',
) -> HardDataset:
    """
    Interpret the given sequence of lines as CSV for a HardDataset.

    Each line is a list of state indexes (ints) separated by `sep`.

    Every line should have the same number of values.

    If the first line contains a non-integer value, then the first
    line will be interpreted as a header line.

    If there is no header line, then the values will be interpreted in the
    same order as `rvs` and the number of values on each line should be
    the same as the number of random variables in `rvs`.

    If there is a header line, then it will be interpreted as the order
    of random variables. There must be a column name in the header to match
    each name of the given random variables. Additional columns will be ignored.

    Leading and trailing whitespace is ignored for each field, including header column names.

    As text file (and StringIO) objects are iterable over lines, here is how to read a csv file::

        with open(csv_filename, 'r') as file:
            hard_dataset_from_csv(rvs, file)

    Here is an example to read from a csv string::

        hard_dataset_from_csv(rvs, csv_string.splitlines())


    Args:
        rvs: the random variables for the returned dataset.
        lines: the sequence of lines to interpret, each line is an instance in the dataset.
        weights: the column in the CSV file holding instance weights. Can be either the
            column number (counting from zero) or a column name (requires a header line).
        sep: the string to use to separate values in a line, default is a comma.
            If set to `None`, lines will be split on any consecutive run of whitespace characters
            (including \n \r \t \f and spaces).
        comment: text starting with this will be treated as a comment. Set to '' to disallow comments.

    Returns:
        a HardDataset.

    Raises:
        ValueError: if the lines do not conform to a CSV format.
    """
    rvs: Sequence[RandomVariable] = tuple(rvs)

    # Define `clean_line` being sensitive to comments.
    if len(comment) > 0:
        def clean_line(l: str) -> str:
            i = l.find(comment)
            if i >= 0:
                l = l[:i]
            return l.strip()
    else:
        def clean_line(l: str) -> str:
            return l.strip()

    # Get the first line which may be a header line or data line
    it = iter(lines)
    try:
        while True:
            line = clean_line(next(it))
            if len(line) > 0:
                break
    except StopIteration:
        # Empty dataset with the given random variables
        return HardDataset((rv, []) for rv in rvs)

    values: List[str] = [value.strip() for value in line.split(sep)]
    number_of_columns: int = len(values)
    series: List[List[int]]  # series[dataset-column] = list of values
    weight_series: Optional[List[float]] = None
    column_map: List[int]  # column_map[dataset-column] = input-column
    if all(_is_number(value) for value in values):
        # First line is not a header line
        if weights is None:
            if number_of_columns != len(rvs):
                raise ValueError('number of columns does not match number of random variables')
            column_map = list(range(len(rvs)))
        else:
            if number_of_columns != len(rvs) + 1:
                raise ValueError('number of columns does not match number of random variables and weight column')
            if not isinstance(weights, int):
                raise ValueError('no header detected - `weights` must be a column number')
            if not (-number_of_columns <= weights < number_of_columns):
                raise ValueError('`weights` column number out of range')
            column_map = list(range(len(rvs) + 1))
            column_map.pop(weights)

        # Initialise series with the first line of data
        series = [[int(values[i])] for i in column_map]
        if weights is not None:
            weight_series = [float(values[weights])]

    else:
        # First line is a header line
        # Lookup each random variable to find its column
        column_map = [
            values.index(rv.name)  # will raise ValueError if not found
            for rv in rvs
        ]
        if isinstance(weights, str):
            # Convert weights column name to column number
            weights: int = values.index(weights)  # will raise ValueError if not found
        elif isinstance(weights, int) and not (number_of_columns <= weights < number_of_columns):
            raise ValueError('`weights` column number out of range')

        # Initialise each series as empty
        series = [[] for _ in rvs]
        if weights is not None:
            weight_series = []

    # Read remaining data lines
    for line in it:
        line = clean_line(line)
        if len(line) == 0:
            continue
        if len(values) != number_of_columns:
            raise ValueError('number of values does not match number of columns')
        values = line.split(sep)
        for series_i, i in zip(series, column_map):
            series_i.append(int(values[i]))
        if weights is not None:
            weight_series.append(float(values[weights]))

    # Construct the dataset
    return HardDataset(zip(rvs, series), weights=weight_series)


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
