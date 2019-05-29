# -*- coding: utf-8 -*-

"""Collection of transforms
"""

from enum import Enum

import numpy as np
import librosa
from scipy import signal

from src.earthquake.operators import Transform


class Peaks(Transform):
    def __init__(self, name, q, distance):
        super().__init__(name)
        self.q = q
        self.distance = distance

    def apply(self, x):
        peaks, _ = signal.find_peaks(x, height=np.quantile(x, self.q), distance=self.distance)
        return peaks


class Mfcc(Transform):
    """Mel-frequency cepstral coefficients (MFCCs)
    """
    def __init__(self, name, cid):
        super().__init__(name)
        self.cid = cid

    def apply(self, x):
        mfcc = librosa.feature.mfcc(y=x.astype('float32'))
        return np.abs(mfcc[self.cid])


class SpectralCentroid(Transform):
    def __init__(self):
        super().__init__('CENTR')

    def apply(self, x):
        spec_c = librosa.feature.spectral_centroid(y=x.astype('float32'))
        return np.abs(spec_c[0])


class SpectralBandwidth(Transform):
    def __init__(self):
        super().__init__('BW')

    def apply(self, x):
        spec_bw = librosa.feature.spectral_bandwidth(y=x.astype('float32'))
        return np.abs(spec_bw[0])


class SpectralContrast(Transform):
    def __init__(self):
        super().__init__('CONTR')

    def apply(self, x):
        S = np.abs(librosa.stft(x.astype('float32')))
        contrast = librosa.feature.spectral_contrast(S=S)
        return np.abs(contrast[0])


class SpectralFlatness(Transform):
    def __init__(self):
        super().__init__('FLAT')

    def apply(self, x):
        flatness = librosa.feature.spectral_flatness(y=x.astype('float32'))
        return np.abs(flatness[0])


class SpectralRolloff(Transform):
    def __init__(self):
        super().__init__('RLF')

    def apply(self, x):
        rolloff = librosa.feature.spectral_rolloff(y=x.astype('float32'))
        return np.abs(rolloff[0])


class ZerCrossingRate(Transform):
    def __init__(self):
        super().__init__('ZRATE')

    def apply(self, x):
        rate = librosa.feature.zero_crossing_rate(y=x.astype('float32'))
        return np.abs(rate[0])


class Raw(Transform):
    def __init__(self):
        super().__init__('RAW')

    def apply(self, x):
        return x


class Diff(Transform):
    def __init__(self):
        super().__init__('DIFF')

    def apply(self, x):
        return np.diff(x)


class Diff2(Transform):
    def __init__(self):
        super().__init__('DIFF2')

    def apply(self, x):
        return np.diff(x, n=2)


class Abs(Transform):
    def __init__(self):
        super().__init__('ABS')

    def apply(self, x):
        return np.absolute(x)


class Log(Transform):
    def __init__(self):
        super().__init__('LOG')

    def apply(self, x):
        return np.log(x + abs(np.min(x)) + 1)


class Scale(Transform):
    def __init__(self):
        super().__init__('SCALE')

    def apply(self, x):
        return (x - np.mean(x)) / np.std(x)


class Filter(Transform):
    def __init__(self):
        super().__init__('FILT')

    def apply(self, x):
        return signal.medfilt(x, kernel_size=7)


class Power2(Transform):
    def __init__(self):
        super().__init__('POW2')

    def apply(self, x):
        return x ** 2


class Power3(Transform):
    def __init__(self):
        super().__init__('POW3')

    def apply(self, x):
        return x ** 3


class Welch(Transform):
    def __init__(self):
        super().__init__('WELCH')

    def apply(self, x):
        _, pxx = signal.welch(x)
        return pxx

