# -*- coding: utf-8 -*-

"""Utils collection for feature engineering
"""

import os

import joblib
import pandas as pd

from tqdm import tqdm

from src.earthquake.operators import Slice
import src.earthquake.aggregations as aggr
import src.earthquake.transforms as tr

import config


class Segment(object):
    def __init__(self, path_to_hdf, signal_name, target_name, start, n_rows):
        self.path_to_hdf = path_to_hdf
        self.start = start
        self.n_rows = n_rows
        self.stop = start + n_rows
        self.signal_name = signal_name
        self.target_name = target_name
        self.signal = []
        self.target = None

    def __repr__(self):
        return 'Segment(%s, %s): size=%s, target=%s' % \
               (self.start, self.stop, self.n_rows, self.target)

    def __enter__(self):
        return self.__get_data()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def get_signal(self):
        return self.signal

    def get_target(self):
        return self.target

    def __get_data(self):
        data = pd.read_hdf(self.path_to_hdf, start=self.start, stop=self.stop)

        signal = data[self.signal_name].values
        target = data[self.target_name].iloc[-1]

        return signal, target


def make_feature(path2hdf, signal_name, target_name, feature, n_rows, n_segments, is_target=False, desc='', n_jobs=1):
    feature = joblib.Parallel(n_jobs=n_jobs, verbose=False)(
        joblib.delayed(_make_feature)(
            seg_id,
            path2hdf,
            signal_name,
            target_name,
            feature,
            n_rows,
            is_target
        )
        for seg_id in tqdm(range(n_segments), ncols=100, ascii=True, desc=desc)
    )

    return feature


def _make_feature(seg_id, path2hdf, signal_name, target_name, feature, n_rows, is_target):
    start = seg_id * n_rows

    with Segment(path2hdf, signal_name, target_name, start, n_rows) as segment:
        if is_target:
            x = segment[1]
        else:
            signal = segment[0]

            for operator in feature:
                signal = operator.apply_soft(signal)

            x = signal

        del segment
    return x


def make_pool(slices, transforms, aggregations, **kwargs):
    n_segments = kwargs.get('n_segments', None)
    n_rows_seg = kwargs.get('n_rows_seg', None)
    path2hdf = kwargs.get('path2hdf', None)
    path2csv = kwargs.get('path2csv', None)
    append = kwargs.get('append', False)
    n_jobs = kwargs.get('n_jobs', 1)
    signal_name = kwargs.get('signal_name', None)
    target_name = kwargs.get('target_name', None)
    n_feat_to_csv = kwargs.get('n_feat_to_csv', 1)

    data = pd.DataFrame()

    if append:
        if path2csv is not None:
            if os.path.isfile(path2csv):
                data = pd.read_csv(path2csv)

    if 'seg_id' not in data.columns:
        data['seg_id'] = [i for i in range(n_segments)]

    if 'target' not in data.columns:
        data['target'] = make_feature(
            path2hdf, signal_name, target_name, None, n_rows_seg, n_segments,
            is_target=True, desc='target', n_jobs=n_jobs)

    iter_id = 0

    for s in slices:
        for t in transforms:
            for a in aggregations:
                feature = [s, t, a]
                feature_name = '%s_%s_%s' % (s.get_name(), t.get_name(), a.get_name())
                feature_name = feature_name.lower()

                if feature_name not in data.columns:
                    data[feature_name] = make_feature(
                        path2hdf, signal_name, target_name, feature, n_rows_seg, n_segments,
                        is_target=False, desc=feature_name, n_jobs=n_jobs)

                iter_id += 1

                if iter_id % n_feat_to_csv == 0:
                    if path2csv is not None:
                        data.to_csv(path2csv, index=False, float_format='%.5f')

    if path2csv is not None:
        data.to_csv(path2csv, index=False, float_format='%.5f')


def init_slices():
    slices = [
        Slice('all'),
        Slice('last1000', config.n_rows_all - 1000, config.n_rows_all)
    ]

#    for i, n_obs in enumerate(config.slice_counts):
#        slices += [Slice('FIRST' + str(n_obs // 1000), i + 2, 0, n_obs)]
#        slices += [Slice('LAST' + str(n_obs // 1000), i + len(config.slice_counts) + 2,
#                         config.n_rows_all - n_obs, config.n_rows_all)]

    return slices


def init_aggregations():
#    aggregations = [
#        aggr.Average(), aggr.StDev(), aggr.Max(), aggr.Min(), aggr.MaxMin(), aggr.MinMax(),
#        aggr.Median(), aggr.Argmax(), aggr.Argmin(), aggr.Skew(), aggr.Kurtosis(),
#        aggr.WelchFreq(), aggr.WelchFreqAvg(), aggr.WelchDensityAvg()]

#    max_index = get_max_index(aggregations)
#
#    for i, q in enumerate(config.quantiles):
#        aggregations += [aggr.Quantile('q' + str(q)[2:], max_index + i + 1, q)]
#
#    max_index = get_max_index(aggregations)
#
#    for i, q in enumerate(config.peak_quantiles):
#        aggregations += [
#            aggr.NPeaks('npeaks' + str(q)[2:], max_index + i + 1, q, distance=config.peak_distance)]

    aggregations = [
        aggr.Average()
    ]

    return aggregations


def init_transforms():
    transforms = [
        tr.SpectralCentroid(), tr.SpectralBandwidth(), tr.SpectralContrast(), tr.SpectralFlatness(),
        tr.SpectralRolloff(), tr.ZerCrossingRate()]

    for i in range(20):
        transforms += [tr.Mfcc('mfcc' + str(i), i)]

    return transforms


def main():
    slices = init_slices()
    aggregations = init_aggregations()
    transforms = init_transforms()

    make_pool(slices, transforms, aggregations,
              path2hdf=config.path_to_hdf,
              path2csv=config.path_to_train,
              signal_name=config.signal_name,
              target_name=config.target_name,
              n_segments=config.n_segments,
              n_rows_seg=config.n_rows_seg,
              n_jobs=config.n_jobs,
              append=False,
              n_feat_to_csv=10)


if __name__ == '__main__':
    main()



