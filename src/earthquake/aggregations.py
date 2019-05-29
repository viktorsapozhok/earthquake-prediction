# -*- coding: utf-8 -*-

"""Collection of aggregations
"""

from enum import Enum

import numpy as np
from scipy import signal, stats

from src.earthquake.operators import Aggregation


class Aggr(Enum):
    AVG = 0
    STD = 1
    MAX = 2
    MAXMIN = 3
    MIN = 4
    MINMAX = 5
    MED = 6
    ARGMAX = 7
    ARGMIN = 8
    SKEW = 9
    KURT = 10
    WF = 11
    WFAVG = 12
    WDAVG = 13


class Quantile(Aggregation):
    def __init__(self, name, index, q):
        super().__init__(name=name, index=index)
        self.q = q

    def apply(self, x):
        return np.quantile(x, self.q)


class NPeaks(Aggregation):
    def __init__(self, name, index, q, distance):
        super().__init__(name=name, index=index)
        self.q = q
        self.distance = distance

    def apply(self, x):
        peaks, _ = signal.find_peaks(x, height=np.quantile(x, self.q), distance=self.distance)
        return len(peaks)


class Average(Aggregation):
    def __init__(self):
        super().__init__(aggr=Aggr.AVG)

    def apply(self, x):
        return np.mean(x)


class StDev(Aggregation):
    def __init__(self):
        super().__init__(aggr=Aggr.STD)

    def apply(self, x):
        return np.std(x)


class Max(Aggregation):
    def __init__(self):
        super().__init__(aggr=Aggr.MAX)

    def apply(self, x):
        return np.max(x)


class MaxMin(Aggregation):
    def __init__(self):
        super().__init__(aggr=Aggr.MAXMIN)

    def apply(self, x):
        return np.max(x) + np.min(x)


class Min(Aggregation):
    def __init__(self):
        super().__init__(aggr=Aggr.MIN)

    def apply(self, x):
        return np.min(x)


class Median(Aggregation):
    def __init__(self):
        super().__init__(aggr=Aggr.MED)

    def apply(self, x):
        return np.median(x)


class MinMax(Aggregation):
    def __init__(self):
        super().__init__(aggr=Aggr.MINMAX)

    def apply(self, x):
        return np.max(x) - np.min(x)


class Argmax(Aggregation):
    def __init__(self):
        super().__init__(aggr=Aggr.ARGMAX)

    def apply(self, x):
        return np.argmax(x)


class Argmin(Aggregation):
    def __init__(self):
        super().__init__(aggr=Aggr.ARGMIN)

    def apply(self, x):
        return np.argmin(x)


class Skew(Aggregation):
    def __init__(self):
        super().__init__(aggr=Aggr.SKEW)

    def apply(self, x):
        return stats.skew(x)


class Kurtosis(Aggregation):
    def __init__(self):
        super().__init__(aggr=Aggr.KURT)

    def apply(self, x):
        return stats.kurtosis(x)


class WelchFreq(Aggregation):
    def __init__(self):
        super().__init__(aggr=Aggr.WF)

    def apply(self, x):
        f, pxx = signal.welch(x)
        return f[np.argmax(pxx)]


class WelchFreqAvg(Aggregation):
    def __init__(self):
        super().__init__(aggr=Aggr.WFAVG)

    def apply(self, x):
        f, pxx = signal.welch(x)
        return np.average(f, weights=pxx)


class WelchDensityAvg(Aggregation):
    def __init__(self):
        super().__init__(aggr=Aggr.WDAVG)

    def apply(self, x):
        f, pxx = signal.welch(x)
        return np.average(pxx, weights=f)
