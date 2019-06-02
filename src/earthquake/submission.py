# -*- coding: utf-8 -*-

"""Make prediction and prepare results for submission
"""

import joblib
import pandas as pd

from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from src.earthquake.operators import Slice
import src.earthquake.aggregations as aggr
import src.earthquake.transforms as tr

import config


def init_features():
    s_all = Slice('all')
    mfcc_4 = tr.Mfcc('mfcc4', 4)
    mfcc_5 = tr.Mfcc('mfcc5', 5)
    mfcc_15 = tr.Mfcc('mfcc15', 15)
    rollstd10 = tr.RollStd('rollstd10', 10)
    rollstd100 = tr.RollStd('rollstd100', 100)
    rollstd1000 = tr.RollStd('rollstd1000', 1000)
    avg = aggr.Average()
    q5 = aggr.Quantile('q5', 0.05)

    features = [
        [s_all, mfcc_4, avg], [s_all, mfcc_5, avg], [s_all, mfcc_15, avg],
        [s_all, mfcc_4, q5], [s_all, mfcc_5, q5], [s_all, mfcc_15, q5],
        [s_all, rollstd10, q5], [s_all, rollstd100, q5], [s_all, rollstd1000, q5],
        [s_all, rollstd10, avg], [s_all, rollstd100, avg], [s_all, rollstd1000, avg]
    ]

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
    test_set.to_csv(config.path_to_test_set, index=False, float_format='%.5f')

    data = pd.read_csv(config.path_to_train)

    data['target_int'] = data['target'].round().astype(int)
    model = RandomForestClassifier(n_estimators=100, max_features=1, min_samples_leaf=10)
    model.fit(data[feature_names], data['target_int'])
    data['y_hat_int'] = model.predict(data[feature_names])

    model = RandomForestRegressor(n_estimators=100)

    train = data.query('abs(target_int - y_hat_int) < 6').reset_index()
    model.fit(train[feature_names], train['target'])

    results = pd.DataFrame()
    results['seg_id'] = test_set['seg_id']
    results[config.target_name] = model.predict(test_set[feature_names])

    results.to_csv(config.path_to_results, index=False, float_format='%.5f')


if __name__ == '__main__':
    submit()

