import argparse
from itertools import product
import warnings

from joblib import Parallel, delayed
import librosa
import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from tsfresh.feature_extraction import feature_calculators

from earthquake import config

warnings.filterwarnings("ignore")


class FeatureGenerator(object):
    """Feature engineering.
    """

    def __init__(
            self,
            path_to_store,
            is_train=True,
            n_rows=1e6,
            n_jobs=1,
            segment_size=150000
    ):
        """Decomposition of initial signal into the set of features.

        Args:
            path_to_store:
                Path to .hdf store with original signal data.
            is_train:
                True, if creating the training set.
            n_rows:
                Amount of rows in training store.
            n_jobs:
                Amount of parallel jobs.
            segment_size:
                Amount of observations in each segment
        """

        self.path_to_store = path_to_store
        self.n_rows = n_rows
        self.n_jobs = n_jobs
        self.segment_size = segment_size
        self.is_train = is_train

        if self.is_train:
            self.total = int(self.n_rows / self.segment_size)
            self.store = None
            self.keys = None
        else:
            self.store = pd.HDFStore(self.path_to_store, mode='r')
            self.keys = self.store.keys()
            self.total = len(self.keys)

    def __del__(self):
        if self.store is not None:
            self.store.close()

    def segments(self):
        """Returns generator object to iterate over segments.
        """

        if self.is_train:
            for i in range(self.total):
                start = i * self.segment_size
                stop = (i + 1) * self.segment_size

                # read one segment of data from .hdf store
                data = pd.read_hdf(self.path_to_store, start=start, stop=stop)

                x = data['acoustic_data'].values
                y = data['time_to_failure'].values[-1]
                seg_id = 'train_' + str(i)

                del data
                yield seg_id, x, y
        else:
            for key in self.keys:
                seg_id = key[1:]
                x = self.store[key]['acoustic_data'].values
                yield seg_id, x, -999

    def get_features(self, x, y, seg_id):
        x = pd.Series(x)

        # fast fourier transform
        zc = np.fft.fft(x)
        # real part
        realFFT = pd.Series(np.real(zc))
        # imaginary part
        imagFFT = pd.Series(np.imag(zc))

        main_dict = self.features(x, y, seg_id)
        r_dict = self.features(realFFT, y, seg_id)
        i_dict = self.features(imagFFT, y, seg_id)

        for k, v in r_dict.items():
            if k not in ['target', 'seg_id']:
                main_dict[f'fftr_{k}'] = v

        for k, v in i_dict.items():
            if k not in ['target', 'seg_id']:
                main_dict[f'ffti_{k}'] = v

        return main_dict

    def features(self, x, y, seg_id):
        feature_dict = dict()
        feature_dict['target'] = y
        feature_dict['seg_id'] = seg_id

        # lists with parameters to iterate over them
        percentiles = [
            1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]
        hann_windows = [
            50, 150, 1500, 15000]
        spans = [
            300, 3000, 30000, 50000]
        windows = [
            10, 50, 100, 500, 1000, 10000]
        borders = list(range(-4000, 4001, 1000))
        peaks = [
            10, 20, 50, 100]
        coefs = [
            1, 5, 10, 50, 100]
        autocorr_lags = [
            5, 10, 50, 100, 500, 1000, 5000, 10000]

        # basic stats
        feature_dict['mean'] = x.mean()
        feature_dict['std'] = x.std()
        feature_dict['max'] = x.max()
        feature_dict['min'] = x.min()

        # basic stats on absolute values
        feature_dict['mean_change_abs'] = np.mean(np.diff(x))
        feature_dict['abs_max'] = np.abs(x).max()
        feature_dict['abs_mean'] = np.abs(x).mean()
        feature_dict['abs_std'] = np.abs(x).std()

        # geometric and harmonic means
        feature_dict['hmean'] = stats.hmean(np.abs(x[np.nonzero(x)[0]]))
        feature_dict['gmean'] = stats.gmean(np.abs(x[np.nonzero(x)[0]]))

        # k-statistic and moments
        for i in range(1, 5):
            feature_dict[f'kstat_{i}'] = stats.kstat(x, i)
            feature_dict[f'moment_{i}'] = stats.moment(x, i)

        for i in [1, 2]:
            feature_dict[f'kstatvar_{i}'] = stats.kstatvar(x, i)

        # aggregations on various slices of data
        for agg_type, slice_length, direction in product(
                ['std', 'min', 'max', 'mean'],
                [1000, 10000, 50000],
                ['first', 'last']):
            if direction == 'first':
                feature_dict[f'{agg_type}_{direction}_{slice_length}'] = \
                    x[:slice_length].agg(agg_type)
            elif direction == 'last':
                feature_dict[f'{agg_type}_{direction}_{slice_length}'] = \
                    x[-slice_length:].agg(agg_type)

        feature_dict['max_to_min'] = x.max() / np.abs(x.min())
        feature_dict['max_to_min_diff'] = x.max() - np.abs(x.min())
        feature_dict['count_big'] = len(x[np.abs(x) > 500])
        feature_dict['sum'] = x.sum()

        feature_dict['mean_change_rate'] = self.calc_change_rate(x)

        # calc_change_rate on slices of data
        for slice_length, direction in product(
                [1000, 10000, 50000], ['first', 'last']):
            if direction == 'first':
                feature_dict[f'mean_change_rate_{direction}_{slice_length}'] = \
                    self.calc_change_rate(x[:slice_length])
            elif direction == 'last':
                feature_dict[f'mean_change_rate_{direction}_{slice_length}'] = \
                    self.calc_change_rate(x[-slice_length:])

        # percentiles on original and absolute values
        for p in percentiles:
            feature_dict[f'percentile_{p}'] = np.percentile(x, p)
            feature_dict[f'abs_percentile_{p}'] = np.percentile(np.abs(x), p)

        feature_dict['trend'] = self.add_trend_feature(x)
        feature_dict['abs_trend'] = self.add_trend_feature(x, abs_values=True)

        feature_dict['mad'] = x.mad()
        feature_dict['kurt'] = x.kurtosis()
        feature_dict['skew'] = x.skew()
        feature_dict['med'] = x.median()

        feature_dict['Hilbert_mean'] = np.abs(signal.hilbert(x)).mean()

        for hw in hann_windows:
            feature_dict[f'Hann_window_mean_{hw}'] = \
                (signal.convolve(x, signal.hann(hw), mode='same') / sum(signal.hann(hw))).mean()

        feature_dict['classic_sta_lta1_mean'] = \
            self.classic_sta_lta(x, 500, 10000).mean()
        feature_dict['classic_sta_lta2_mean'] = \
            self.classic_sta_lta(x, 5000, 100000).mean()
        feature_dict['classic_sta_lta3_mean'] = \
            self.classic_sta_lta(x, 3333, 6666).mean()
        feature_dict['classic_sta_lta4_mean'] = \
            self.classic_sta_lta(x, 10000, 25000).mean()
        feature_dict['classic_sta_lta5_mean'] = \
            self.classic_sta_lta(x, 50, 1000).mean()
        feature_dict['classic_sta_lta6_mean'] = \
            self.classic_sta_lta(x, 100, 5000).mean()
        feature_dict['classic_sta_lta7_mean'] = \
            self.classic_sta_lta(x, 333, 666).mean()
        feature_dict['classic_sta_lta8_mean'] = \
            self.classic_sta_lta(x, 4000, 10000).mean()

        # exponential rolling statistics
        ewma = pd.Series.ewm
        for s in spans:
            feature_dict[f'exp_Moving_average_{s}_mean'] = \
                (ewma(x, span=s).mean(skipna=True)).mean(skipna=True)
            feature_dict[f'exp_Moving_average_{s}_std'] = \
                (ewma(x, span=s).mean(skipna=True)).std(skipna=True)
            feature_dict[f'exp_Moving_std_{s}_mean'] = \
                (ewma(x, span=s).std(skipna=True)).mean(skipna=True)
            feature_dict[f'exp_Moving_std_{s}_std'] = \
                (ewma(x, span=s).std(skipna=True)).std(skipna=True)

        feature_dict['iqr'] = np.subtract(*np.percentile(x, [75, 25]))
        feature_dict['iqr1'] = np.subtract(*np.percentile(x, [95, 5]))
        feature_dict['ave10'] = stats.trim_mean(x, 0.1)

        for slice_length, threshold in product(
                [50000, 100000, 150000], [5, 10, 20, 50, 100]):
            feature_dict[f'count_big_{slice_length}_threshold_{threshold}'] = \
                (np.abs(x[-slice_length:]) > threshold).sum()
            feature_dict[f'count_big_{slice_length}_less_threshold_{threshold}'] = \
                (np.abs(x[-slice_length:]) < threshold).sum()

        feature_dict['range_minf_m4000'] = \
            feature_calculators.range_count(x, -np.inf, -4000)
        feature_dict['range_p4000_pinf'] = \
            feature_calculators.range_count(x, 4000, np.inf)

        for i, j in zip(borders, borders[1:]):
            feature_dict[f'range_{i}_{j}'] = feature_calculators.range_count(x, i, j)

        for autocorr_lag in autocorr_lags:
            feature_dict[f'autocorrelation_{autocorr_lag}'] = \
                feature_calculators.autocorrelation(x, autocorr_lag)
            feature_dict[f'c3_{autocorr_lag}'] = \
                feature_calculators.c3(x, autocorr_lag)

        for p in percentiles:
            feature_dict[f'binned_entropy_{p}'] = \
                feature_calculators.binned_entropy(x, p)

        feature_dict['num_crossing_0'] = \
            feature_calculators.number_crossing_m(x, 0)

        for peak in peaks:
            feature_dict[f'num_peaks_{peak}'] = feature_calculators.number_peaks(x, peak)

        for c in coefs:
            feature_dict[f'spkt_welch_density_{c}'] = \
                list(feature_calculators.spkt_welch_density(x, [{'coeff': c}]))[0][1]
            feature_dict[f'time_rev_asym_stat_{c}'] = \
                feature_calculators.time_reversal_asymmetry_statistic(x, c)

        for w in windows:
            x_roll_std = x.rolling(w).std().dropna().values
            x_roll_mean = x.rolling(w).mean().dropna().values

            feature_dict[f'ave_roll_std_{w}'] = x_roll_std.mean()
            feature_dict[f'std_roll_std_{w}'] = x_roll_std.std()
            feature_dict[f'max_roll_std_{w}'] = x_roll_std.max()
            feature_dict[f'min_roll_std_{w}'] = x_roll_std.min()

            for p in percentiles:
                feature_dict[f'percentile_roll_std_{p}_window_{w}'] = \
                    np.percentile(x_roll_std, p)

            feature_dict[f'av_change_abs_roll_std_{w}'] = \
                np.mean(np.diff(x_roll_std))
            feature_dict[f'av_change_rate_roll_std_{w}'] = \
                np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
            feature_dict[f'abs_max_roll_std_{w}'] = \
                np.abs(x_roll_std).max()

            feature_dict[f'ave_roll_mean_{w}'] = x_roll_mean.mean()
            feature_dict[f'std_roll_mean_{w}'] = x_roll_mean.std()
            feature_dict[f'max_roll_mean_{w}'] = x_roll_mean.max()
            feature_dict[f'min_roll_mean_{w}'] = x_roll_mean.min()

            for p in percentiles:
                feature_dict[f'percentile_roll_mean_{p}_window_{w}'] = \
                    np.percentile(x_roll_mean, p)

            feature_dict[f'av_change_abs_roll_mean_{w}'] = \
                np.mean(np.diff(x_roll_mean))
            feature_dict[f'av_change_rate_roll_mean_{w}'] = \
                np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
            feature_dict[f'abs_max_roll_mean_{w}'] = \
                np.abs(x_roll_mean).max()

        # Mel-frequency cepstral coefficients (MFCCs)
        x = x.values.astype('float32')
        mfcc = librosa.feature.mfcc(y=x)
        for i in range(len(mfcc)):
            feature_dict[f'mfcc_{i}_avg'] = np.mean(np.abs(mfcc[i]))

        # spectral features
        feature_dict['spectral_centroid'] = \
            np.mean(np.abs(librosa.feature.spectral_centroid(y=x)[0]))
        feature_dict['zero_crossing_rate'] = \
            np.mean(np.abs(librosa.feature.zero_crossing_rate(y=x)[0]))
        feature_dict['spectral_flatness'] = \
            np.mean(np.abs(librosa.feature.spectral_flatness(y=x)[0]))
        feature_dict['spectral_contrast'] = \
            np.mean(np.abs(librosa.feature.spectral_contrast(S=np.abs(librosa.stft(x)))[0]))
        feature_dict['spectral_bandwidth'] = \
            np.mean(np.abs(librosa.feature.spectral_bandwidth(y=x)[0]))

        return feature_dict

    def generate(self):
        feature_list = []
        res = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(self.get_features)(x, y, s)
            for s, x, y in tqdm(self.segments(),
                                total=self.total,
                                ncols=100,
                                desc='generating features',
                                ascii=True))
        for r in res:
            feature_list.append(r)

        return pd.DataFrame(feature_list)

    @staticmethod
    def add_trend_feature(arr, abs_values=False):
        idx = np.array(range(len(arr)))

        if abs_values:
            arr = np.abs(arr)

        lr = LinearRegression()
        lr.fit(idx.reshape(-1, 1), arr)

        return lr.coef_[0]

    @staticmethod
    def classic_sta_lta(x, length_sta, length_lta):
        sta = np.cumsum(x ** 2)

        # Convert to float
        sta = np.require(sta, dtype=np.float)

        # Copy for LTA
        lta = sta.copy()

        # Compute the STA and the LTA
        sta[length_sta:] = sta[length_sta:] - sta[:-length_sta]
        sta /= length_sta
        lta[length_lta:] = lta[length_lta:] - lta[:-length_lta]
        lta /= length_lta

        # Pad zeros
        sta[:length_lta - 1] = 0

        # Avoid division by zero by setting zero values to tiny float
        dtiny = np.finfo(0.0).tiny
        idx = lta < dtiny
        lta[idx] = dtiny

        return sta / lta

    @staticmethod
    def calc_change_rate(x):
        change = (np.diff(x) / x[:-1]).values
        change = change[np.nonzero(change)[0]]
        change = change[~np.isnan(change)]
        change = change[change != -np.inf]
        change = change[change != np.inf]
        return np.mean(change)


def main(args):
    if args['train']:
        fg = FeatureGenerator(
            config.path_to_train_store,
            is_train=True, n_rows=config.n_rows_all,
            n_jobs=config.n_jobs, segment_size=config.segment_size)
    else:
        fg = FeatureGenerator(
            config.path_to_test_store,
            is_train=False, n_jobs=config.n_jobs)

    data = fg.generate()
    data.to_csv(config.path_to_test, index=False, float_format='%.5f')


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='features generator',
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument(
        '--train', action='store_true',
        help='make train set')

    arg_parser.add_argument(
        '--test', action='store_true',
        help='make test set')

    main(vars(arg_parser.parse_args()))
