"""Distributed operations."""

import abc

from parallelism.sharding import Sharding


class Operation(metaclass=abc.ABCMeta):
    def __init__(self, sharding: Sharding) -> None:
        self._sharding = sharding

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass
