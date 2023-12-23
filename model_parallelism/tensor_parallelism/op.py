"""Distributed operations."""

import abc

from sharding import Sharding


class Op(metaclass=abc.ABCMeta):
    def __init__(self, sharding: Sharding):
        self._sharding = sharding

    @abc.abstractmethod
    def __call__(self, *args, device, **kwargs):
        pass
