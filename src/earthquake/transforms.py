# -*- coding: utf-8 -*-

"""Collection of transforms
"""

from enum import Enum

import numpy as np
import librosa
from scipy import signal

from src.earthquake.operators import Transform


class Tran(Enum):
    RAW = 100
    DIFF = 101
    DIFF2 = 102
    ABS = 103
    LOG = 104
    SCALE = 105
    FILT = 106
    POW2 = 107
    POW3 = 108
    WELCH = 109
    CENTR = 110
    BW = 111
    CONTR = 112
    FLAT = 113
    RLF = 114
    ZRATE = 115


class Peaks(Transform):
    def __init__(self, name, index, q, distance):
        super().__init__(name, index)
        self.q = q
        self.distance = distance

    def apply(self, x):
        peaks, _ = signal.find_peaks(x, height=np.quantile(x, self.q), distance=self.distance)
        return peaks


class Mfcc(Transform):
    """Mel-frequency cepstral coefficients (MFCCs)
    """
    def __init__(self, name, index, cid):
        super().__init__(name, index)
        self.cid = cid

    def apply(self, x):
        mfcc = librosa.feature.mfcc(y=x.astype('float32'))
        return np.abs(mfcc[self.cid])


class SpectralCentroid(Transform):
    def __init__(self):
        super().__init__(tran=Tran.CENTR)

    def apply(self, x):
        spec_c = librosa.feature.spectral_centroid(y=x.astype('float32'))
        return np.abs(spec_c[0])


class SpectralBandwidth(Transform):
    def __init__(self):
        super().__init__(tran=Tran.BW)

    def apply(self, x):
        spec_bw = librosa.feature.spectral_bandwidth(y=x.astype('float32'))
        return np.abs(spec_bw[0])


class SpectralContrast(Transform):
    def __init__(self):
        super().__init__(tran=Tran.CONTR)

    def apply(self, x):
        S = np.abs(librosa.stft(x.astype('float32')))
        contrast = librosa.feature.spectral_contrast(S=S)
        return np.abs(contrast[0])


class SpectralFlatness(Transform):
    def __init__(self):
        super().__init__(tran=Tran.FLAT)

    def apply(self, x):
        flatness = librosa.feature.spectral_flatness(y=x.astype('float32'))
        return np.abs(flatness[0])


class SpectralRolloff(Transform):
    def __init__(self):
        super().__init__(tran=Tran.RLF)

    def apply(self, x):
        rolloff = librosa.feature.spectral_rolloff(y=x.astype('float32'))
        return np.abs(rolloff[0])


class ZerCrossingRate(Transform):
    def __init__(self):
        super().__init__(tran=Tran.ZRATE)

    def apply(self, x):
        rate = librosa.feature.zero_crossing_rate(y=x.astype('float32'))
        return np.abs(rate[0])


class Raw(Transform):
    def __init__(self):
        super().__init__(tran=Tran.RAW)

    def apply(self, x):
        return x


class Diff(Transform):
    def __init__(self):
        super().__init__(tran=Tran.DIFF)

    def apply(self, x):
        return np.diff(x)


class Diff2(Transform):
    def __init__(self):
        super().__init__(tran=Tran.DIFF2)

    def apply(self, x):
        return np.diff(x, n=2)


class Abs(Transform):
    def __init__(self):
        super().__init__(tran=Tran.ABS)

    def apply(self, x):
        return np.absolute(x)


class Log(Transform):
    def __init__(self):
        super().__init__(tran=Tran.LOG)

    def apply(self, x):
        return np.log(x + abs(np.min(x)) + 1)


class Scale(Transform):
    def __init__(self):
        super().__init__(tran=Tran.SCALE)

    def apply(self, x):
        return (x - np.mean(x)) / np.std(x)


class Filter(Transform):
    def __init__(self):
        super().__init__(tran=Tran.FILT)

    def apply(self, x):
        return signal.medfilt(x, kernel_size=7)


class Power2(Transform):
    def __init__(self):
        super().__init__(tran=Tran.POW2)

    def apply(self, x):
        return x ** 2


class Power3(Transform):
    def __init__(self):
        super().__init__(tran=Tran.POW3)

    def apply(self, x):
        return x ** 3


class Welch(Transform):
    def __init__(self):
        super().__init__(tran=Tran.WELCH)

    def apply(self, x):
        _, pxx = signal.welch(x)
        return pxx

