"""
A module with extra randomisation functions.
"""
import random
from typing import Protocol, Sequence, Tuple, Any, List


class Random(Protocol):
    """
    A minimum protocol for a random number generator as used by CK.
    The usual `random` package implements this protocol.
    """

    def random(self) -> float:
        """
        Returns a random float in the interval [0, 1), includes 0, excludes 1.
        """
        ...

    def randrange(self, a: int, b: int) -> int:
        """
        Returns a random integer in interval [a, b), includes `a`, excludes `b`.
        """
        ...


def random_pair(size: int, rand: Random = random) -> Tuple[int, int]:
    """
    Return a random pair (i, j) where:
    0 <= i < size,
    0 <= j < size,
    i != j.
    """
    i = rand.randrange(0, size)
    j = (i + rand.randrange(1, size)) % size
    return i, j


def random_permute(items: List[Any], rand: Random = random) -> None:
    """
    Randomly permute the given items.
    For a list of length `n`, this method calls rand.randrange(...) `n - 1` times.

    There is a numpy method to do this, but it uses its own random number generator.
    """
    for i in range(len(items) - 1, 0, -1):  # i = n - 1 down to 1
        j = rand.randrange(0, i + 1)  # 0 <= j <= i
        items[i], items[j] = items[j], items[i]  # exchange


def random_permutation(size: int, rand: Random = random) -> Sequence[int]:
    """
    Return a random permutation of the given size.

    The returned list contains each integer 0 to size - 1
    in a random order.

    This method calls rand.randrange(...) 'size' times.

    There is a numpy method to do this, but it uses its own random number generator.
    """
    result = list(range(size))
    random_permute(result, rand)
    return result
