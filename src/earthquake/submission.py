# -*- coding: utf-8 -*-

"""Make prediction and prepare results for submission
"""

from catboost import CatBoostRegressor
import pandas as pd

import config


def init_features():
    features = [
        'ffti_av_change_rate_roll_mean_1000', 'percentile_roll_std_30_window_50', 'skew',
        'percentile_roll_std_10_window_100', 'percentile_roll_std_30_window_50',
        'percentile_roll_std_20_window_1000', 'ffti_exp_Moving_average_30000_mean',
        'range_3000_4000', 'max_last_10000', 'mfcc_4_avg',
        'fftr_percentile_roll_std_80_window_10000', 'percentile_roll_std_1_window_100',
        'ffti_abs_trend', 'av_change_abs_roll_mean_50', 'mfcc_15_avg'
    ]

    return features


def submit():
    features = init_features()

    train_set = pd.read_csv(config.path_to_train)
    test_set = pd.read_csv(config.path_to_test)

    model = CatBoostRegressor(verbose=False)
    model.fit(train_set[features], train_set['target'])

    results = pd.DataFrame()
    results['seg_id'] = test_set['seg_id']
    results['time_to_failure'] = model.predict(test_set[features])

    results.to_csv(config.path_to_results, index=False, float_format='%.5f')


if __name__ == '__main__':
    submit()

