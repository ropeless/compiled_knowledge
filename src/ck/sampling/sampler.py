from abc import ABC, abstractmethod
from itertools import islice
from typing import Sequence, Iterator

from ck.pgm import RandomVariable, Instance, Indicator


class Sampler(ABC):
    """
    A Sampler provides an unlimited series of samples for one or more random variables.
    The random variables being sampled are provided as a tuple via the `rvs` property.

    A Sampler will either iterate over Instance objects, where each instance is co-indexed
    with `self.rvs`, or may iterate over single state indexes. Whether a Sampler iterates
    over Instance objects or single state indexes is determined by the implementation.
    If iterating over single state indexes, then `len(self.rvs) == 1`.
    """
    __slots__ = ('_rvs', '_condition')

    def __init__(self, rvs: Sequence[RandomVariable], condition: Sequence[Indicator]):
        """
        Args:
            rvs: a collection of the random variables being
                sampled, co-indexed with each sample provided by `iter(self)`.
            condition: condition on `rvs` that are compiled into the sampler.
        """
        self._rvs: Sequence[RandomVariable] = tuple(rvs)
        self._condition: Sequence[Indicator] = tuple(condition)

    @property
    def rvs(self) -> Sequence[RandomVariable]:
        """
        What random variables are being sampled.

        Returns:
            the random variables being sampled, co-indexed with each sample from `iter(self)`.
        """
        return self._rvs

    @property
    def condition(self) -> Sequence[Indicator]:
        """
        Condition on `self.rvs` that are compiled into the sampler.
        """
        return self._condition

    @abstractmethod
    def __iter__(self) -> Iterator[Instance] | Iterator[int]:
        """
        An unlimited series of samples from a random process.
        Each sample is co-indexed with the random variables provided by `self.rvs`.
        """
        ...

    def take(self, number_of_samples: int) -> Iterator[Instance] | Iterator[int]:
        """
        Take a limited number of samples from `iter(self)`.

        Args:
            number_of_samples: a limit on the number of samples to provide.
        """
        return islice(self, number_of_samples)
