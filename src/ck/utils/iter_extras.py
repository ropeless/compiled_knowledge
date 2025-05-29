"""
A module with extra iteration functions.
"""
from functools import reduce as _reduce
from itertools import combinations, chain, islice
from operator import mul as _mul
from typing import Iterable, Tuple, Sequence, TypeVar

_T = TypeVar('_T')


def flatten(iterables: Iterable[Iterable[_T]]) -> Iterable[_T]:
    """
    Iterate over the elements of an iterable of iterables.
    """
    return (elem for iterable in iterables for elem in iterable)


def deep_flatten(iterables: Iterable) -> Iterable:
    """
    Iterate over the flattening of nested iterables.
    """
    for el in iterables:
        if isinstance(el, Iterable) and not isinstance(el, str):
            for sub in deep_flatten(el):
                yield sub
        else:
            yield el


def combos(list_of_lists: Sequence[Sequence[_T]], flip=False) -> Iterable[Tuple[_T, ...]]:
    """
    Iterate over all combinations of taking one element from each of the lists.

    The order of results has the first element changing most rapidly.
    For example, given [[1,2,3],[4,5],[6,7]], combos yields the following:
        (1,4,6), (2,4,6), (3,4,6), (1,5,6), (2,5,6), (3,5,6),
        (1,4,7), (2,4,7), (3,4,7), (1,5,7), (2,5,7), (3,5,7).

    If flip, then the last changes most rapidly.
    """
    num = len(list_of_lists)
    if num == 0:
        yield ()
        return
    rng = range(num)
    indexes = [0] * num
    if flip:
        start = num - 1
        inc = -1
        end = -1
    else:
        start = 0
        inc = 1
        end = num
    while True:
        yield tuple(list_of_lists[i][indexes[i]] for i in rng)
        i = start
        while True:
            indexes[i] += 1
            if indexes[i] < len(list_of_lists[i]):
                break
            indexes[i] = 0
            i += inc
            if i == end:
                return


def combos_ranges(list_of_lens: Sequence[int], flip=False) -> Iterable[Tuple[int, ...]]:
    """
    Equivalent to combos([range(l) for l in list_of_lens], flip).

    The order of results has the first element changing most rapidly.
    If flip, then the last changes most rapidly.
    """
    num = len(list_of_lens)
    if num == 0:
        yield ()
        return
    indexes = [0] * num
    if flip:
        start = num - 1
        inc = -1
        end = -1
    else:
        start = 0
        inc = 1
        end = num
    while True:
        yield tuple(indexes)
        i = start
        while True:
            indexes[i] += 1
            if indexes[i] < list_of_lens[i]:
                break
            indexes[i] = 0
            i += inc
            if i == end:
                return


def pairs(elements: Iterable[_T]) -> Iterable[Tuple[_T, _T]]:
    """
    Iterate over all possible pairs in the given list of elements.
    """
    return combinations(elements, 2)


def sequential_pairs(elements: Sequence[_T]) -> Iterable[Tuple[_T, _T]]:
    """
    Iterate over sequential pairs in the given list of elements.
    """
    for i in range(len(elements) - 1):
        yield elements[i], elements[i + 1]


def powerset(iterable: Iterable[_T], min_size: int = 0, max_size: int = None) -> Iterable[Tuple[_T, ...]]:
    """
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)
    """
    if not isinstance(iterable, (list, tuple)):
        iterable = list(iterable)
    if min_size is None:
        min_size = 0
    if max_size is None:
        max_size = len(iterable)
    return chain.from_iterable(
        combinations(iterable, size)
        for size in range(min_size, max_size + 1)
    )


def unzip(xs: Iterable[Tuple[_T]]) -> Tuple[Iterable[_T]]:
    """
    Inverse function of zip.

    E.g., a, b, c = unzip(zip(a, b, c))

    Note that the Python type of `a`, `b`, and `c` may not be preserved, only
    the contents, order and length are guaranteed.
    """
    return zip(*xs)


def multiply(items: Iterable[_T], initial: _T = 1) -> _T:
    """
    Return the product of the given items.
    """
    return _reduce(_mul, items, initial)


def first(items: Iterable[_T]) -> _T:
    """
    Return the first element of the iterable.
    """
    return next(iter(items))


def take(iterable: Iterable[_T], n: int) -> Iterable[_T]:
    """
    Take the first n elements of the iterable.
    """
    return islice(iterable, n)
