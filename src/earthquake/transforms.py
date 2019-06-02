# -*- coding: utf-8 -*-

"""Collection of transforms
"""

import numpy as np
import librosa
import pandas as pd
from scipy import signal
from pymssa import MSSA

from src.earthquake.operators import Transform


class Peaks(Transform):
    def __init__(self, name, q, distance):
        super().__init__(name)
        self.q = q
        self.distance = distance

    def apply(self, x):
        peaks, _ = signal.find_peaks(x, height=np.quantile(x, self.q), distance=self.distance)
        return peaks


class SSA(Transform):
    """Singular Spectrum Analysis (SSA decomposition)
    """
    def __init__(self, name, cid):
        super().__init__(name)
        self.cid = cid

    def apply(self, x):
        x = x.astype('float32')
        x -= np.mean(x)
        mssa = MSSA(n_components=11, window_size=64, verbose=False)
        mssa.fit(x)
        return mssa.components_[0, :, self.cid]


class Savgol(Transform):
    """Savitsky-Golay filter
    """
    def __init__(self, name, win_len, polyorder):
        super().__init__(name)
        self.win_len = win_len
        self.polyorder = polyorder

    def apply(self, x):
        return signal.savgol_filter(x, self.win_len, self.polyorder)


class Mfcc(Transform):
    """Mel-frequency cepstral coefficients (MFCCs)
    """
    def __init__(self, name, cid):
        super().__init__(name)
        self.cid = cid

    def apply(self, x):
        mfcc = librosa.feature.mfcc(y=x.astype('float32'))
        return np.abs(mfcc[self.cid])


class OnsetStrength(Transform):
    def __init__(self):
        super().__init__('Onset')

    def apply(self, x):
        return librosa.onset.onset_strength(x.astype('float32'))


class Tempo(Transform):
    def __init__(self):
        super().__init__('Tempo')

    def apply(self, x):
        oenv = librosa.onset.onset_strength(x.astype('float32'))
        tempogram = librosa.feature.tempogram(onset_envelope=oenv, win_length=150)
        return tempogram[1]


class Amplitude(Transform):
    def __init__(self):
        super().__init__('Amp')

    def apply(self, x):
        amp = abs(librosa.stft(x.astype('float32')))
        return amp[0]


class Rms(Transform):
    def __init__(self):
        super().__init__('Rms')

    def apply(self, x):
        S, _ = librosa.magphase(librosa.stft(x.astype('float32')))
        rms = librosa.feature.rms(S=S)
        return rms[0]


class Decompose(Transform):
    def __init__(self, name, cid):
        super().__init__(name)
        self.cid = cid

    def apply(self, x):
        S = np.abs(librosa.stft(x.astype('float32')))
        _, acts = librosa.decompose.decompose(S, n_components=8, tol=0.1)
        return acts[self.cid]


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


class RollStd(Transform):
    def __init__(self, name, window):
        super().__init__(name)
        self.window = window

    def apply(self, x):
        s = pd.Series(x).rolling(self.window).std().dropna().values
        return s


class RollAvg(Transform):
    def __init__(self, name, window):
        super().__init__(name)
        self.window = window

    def apply(self, x):
        s = pd.Series(x).rolling(self.window).mean().dropna().values
        return s


class Butter(Transform):
    def __init__(self):
        super().__init__('BUTTER')

    def apply(self, x):
        order = 5
        fs = 7000
        nyq = 0.5 * fs
        low = 500 / nyq
        high = 1250 / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.lfilter(b, a, x.astype('float32'))


