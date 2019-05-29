# -*- coding: utf-8 -*-

"""Make prediction and prepare results for submission
"""

import joblib
import pandas as pd

from tqdm import tqdm

from catboost import CatBoostRegressor

from src.earthquake.operators import Slice
import src.earthquake.aggregations as aggr
import src.earthquake.transforms as tr

import config


def init_features():
    s_all = Slice('all')
    s_last = Slice('last1000', config.n_rows_all - 1000, config.n_rows_all)
    mfcc_0 = tr.Mfcc('mfcc0', 0)
    mfcc_2 = tr.Mfcc('mfcc2', 2)
    mfcc_4 = tr.Mfcc('mfcc4', 4)
    mfcc_5 = tr.Mfcc('mfcc5', 5)
    mfcc_15 = tr.Mfcc('mfcc15', 15)
    centr = tr.SpectralCentroid()
    zrate = tr.ZerCrossingRate()
    avg = aggr.Average()

    features = [
        [s_last, mfcc_15, avg], [s_all, mfcc_15, avg], [s_last, mfcc_0, avg],
        [s_all, centr, avg], [s_last, mfcc_2, avg], [s_all, mfcc_2, avg],
        [s_last, mfcc_5, avg], [s_all, mfcc_0, avg], [s_all, mfcc_5, avg],
        [s_last, mfcc_4, avg], [s_all, mfcc_4, avg], [s_last, centr, avg],
        [s_last, zrate, avg]]

    return features


def get_feature_name(feature):
    return '_'.join([operator.get_name().lower() for operator in feature])


def make_test_set(path_to_test, features, signal_name, n_jobs=1):
    res = {}

    with pd.HDFStore(path_to_test, mode='r') as store:
        keys = store.keys()
        res['seg_id'] = [key[1:] for key in keys]

        for feature in features:
            feature_name = get_feature_name(feature)

            res[feature_name] = joblib.Parallel(n_jobs=n_jobs, verbose=False)(
                joblib.delayed(_make_feature)(
                    store[key],
                    feature,
                    signal_name
                )
                for key in tqdm(keys, ncols=100, ascii=True, desc=feature_name)
            )

    return pd.DataFrame.from_dict(res)


def _make_feature(segment, feature, signal_name):
    signal = segment[signal_name].values

    for operator in feature:
        signal = operator.apply_soft(signal)

    return signal


def submit():
    features = init_features()
    feature_names = [get_feature_name(feature) for feature in features]
    test_set = make_test_set(config.path_to_test, features, config.signal_name, n_jobs=config.n_jobs)

    data = pd.read_csv(config.path_to_train)
    train_x = data[feature_names]
    train_y = data['target']

    model = CatBoostRegressor(iterations=80, random_seed=0, depth=4, random_strength=0.5,
                              loss_function='RMSE', verbose=False)
    model.fit(train_x, train_y)

    results = pd.DataFrame()
    results['seg_id'] = test_set['seg_id']
    results[config.target_name] = model.predict(test_set[feature_names])

    results.to_csv(config.path_to_results, index=False, float_format='%.5f')


if __name__ == '__main__':
    submit()

