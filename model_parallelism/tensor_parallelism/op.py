"""Distributed operations."""

import abc

from sharding import Sharding


class Op(metaclass=abc.ABCMeta):
    def __init__(self, sharding: Sharding):
        self._sharding = sharding

    def __str__(self):
        return f'{self.__class__.__name__}({self._sharding})'

    @abc.abstractmethod
    def __call__(self, *args, device, **kwargs):
        pass
