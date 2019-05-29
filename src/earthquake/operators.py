# -*- coding: utf-8 -*-

"""Feature engineering using genetic algorithm
"""

import abc
from enum import Enum

import numpy as np


class OperatorType(Enum):
    SLICE = 1
    TRANSFORM = 2
    AGGREGATION = 3


class Operator(abc.ABC):
    def __init__(self, name, operator_type):
        self.name = name
        self.operator_type = operator_type

    def __repr__(self):
        return '%s' % self.name

    def get_name(self):
        return self.name

    def is_slice(self):
        return self.operator_type == OperatorType.SLICE

    def is_transform(self):
        return self.operator_type == OperatorType.TRANSFORM

    def is_aggregation(self):
        return self.operator_type == OperatorType.AGGREGATION

    @abc.abstractmethod
    def apply(self, x):
        raise NotImplementedError()

    @abc.abstractmethod
    def apply_soft(self, x):
        raise NotImplementedError()


class Aggregation(Operator):
    def __init__(self, name):
        super().__init__(name, OperatorType.AGGREGATION)

    @abc.abstractmethod
    def apply(self, x):
        raise NotImplementedError()

    def apply_soft(self, x):
        try:
            return self.apply(x)
        except (ValueError, ZeroDivisionError):
            return np.nan


class Transform(Operator):
    def __init__(self, name):
        super().__init__(name, OperatorType.TRANSFORM)

    @abc.abstractmethod
    def apply(self, x):
        raise NotImplementedError()

    def apply_soft(self, x):
        try:
            return self.apply(x)
        except ValueError:
            return x


class Slice(Operator):
    def __init__(self, name, start=None, end=None):
        super().__init__(name, OperatorType.SLICE)
        self.start = None if start is None else int(start)
        self.end = None if end is None else int(end)

    def apply(self, x):
        if self.start is None:
            return x

        if len(x) < self.end:
            return x

        return x[self.start:self.end]

    def apply_soft(self, x):
        try:
            return self.apply(x)
        except IndexError:
            return x


