# -*- coding: utf-8 -*-

"""Collection of aggregations
"""

import numpy as np
from scipy import signal, stats
from tsfresh.feature_extraction import feature_calculators

from src.earthquake.operators import Aggregation


class Quantile(Aggregation):
    def __init__(self, name, q):
        super().__init__(name)
        self.q = q

    def apply(self, x):
        return np.quantile(x, self.q)


class NPeaks(Aggregation):
    def __init__(self, name, q, distance):
        super().__init__(name)
        self.q = q
        self.distance = distance

    def apply(self, x):
        peaks, _ = signal.find_peaks(x, height=np.quantile(x, self.q), distance=self.distance)
        return len(peaks)


class Average(Aggregation):
    def __init__(self):
        super().__init__('AVG')

    def apply(self, x):
        return np.mean(x)


class StDev(Aggregation):
    def __init__(self):
        super().__init__('STD')

    def apply(self, x):
        return np.std(x)


class Max(Aggregation):
    def __init__(self):
        super().__init__('MAX')

    def apply(self, x):
        return np.max(x)


class MaxMin(Aggregation):
    def __init__(self):
        super().__init__('MAXMIN')

    def apply(self, x):
        return np.max(x) + np.min(x)


class Min(Aggregation):
    def __init__(self):
        super().__init__('MIN')

    def apply(self, x):
        return np.min(x)


class Median(Aggregation):
    def __init__(self):
        super().__init__('MED')

    def apply(self, x):
        return np.median(x)


class MinMax(Aggregation):
    def __init__(self):
        super().__init__('MINMAX')

    def apply(self, x):
        return np.max(x) - np.min(x)


class Argmax(Aggregation):
    def __init__(self):
        super().__init__('ARGMAX')

    def apply(self, x):
        return np.argmax(x)


class Argmin(Aggregation):
    def __init__(self):
        super().__init__('ARGMIN')

    def apply(self, x):
        return np.argmin(x)


class Skew(Aggregation):
    def __init__(self):
        super().__init__('SKEW')

    def apply(self, x):
        return stats.skew(x)


class Kurtosis(Aggregation):
    def __init__(self):
        super().__init__('KURT')

    def apply(self, x):
        return stats.kurtosis(x)


class WelchFreq(Aggregation):
    def __init__(self):
        super().__init__('WF')

    def apply(self, x):
        f, pxx = signal.welch(x)
        return f[np.argmax(pxx)]


class WelchFreqAvg(Aggregation):
    def __init__(self):
        super().__init__('WFAVG')

    def apply(self, x):
        f, pxx = signal.welch(x)
        return np.average(f, weights=pxx)


class WelchDensityAvg(Aggregation):
    def __init__(self):
        super().__init__('WDAVG')

    def apply(self, x):
        f, pxx = signal.welch(x)
        return np.average(pxx, weights=f)


class Slope(Aggregation):
    def __init__(self):
        super().__init__('SLOPE')

    def apply(self, x):
        slope, _, _, _, _ = stats.linregress(np.asarray(range(len(x))), x)
        return slope


class Autocorr(Aggregation):
    def __init__(self):
        super().__init__('AUTOCORR')

    def apply(self, x):
        return feature_calculators.autocorrelation(x.astype('float32'), 5)


class NumPeaks(Aggregation):
    def __init__(self):
        super().__init__('NUMPEAKS')

    def apply(self, x):
        return feature_calculators.number_peaks(x.astype('float32'), 10)
