# -*- coding: utf-8 -*-

"""Make prediction and prepare results for submission
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

import config


def init_features():
    features = [
        'mfcc_15_avg', 'std_roll_mean_100', 'ffti_percentile_roll_std_5_window_10000', 'mfcc_4_avg',
        'fftr_percentile_roll_std_80_window_10000', 'percentile_roll_std_20_window_1000',
        'ffti_exp_Moving_average_30000_mean', 'fftr_time_rev_asym_stat_100',
        'fftr_percentile_roll_std_30_window_100', 'percentile_roll_std_30_window_50',
        'fftr_num_peaks_100', 'ffti_Hann_window_mean_1500', 'fftr_percentile_roll_std_30_window_50',
        'fftr_percentile_roll_std_1_window_50', 'percentile_roll_std_40_window_1000'
    ]

    return features


def submit():
    features = init_features()

    train_set = pd.read_csv(config.path_to_train)
    test_set = pd.read_csv(config.path_to_test_set)

    model = RandomForestRegressor(n_estimators=500)
    model.fit(train_set[features], train_set['target'])

    results = pd.DataFrame()
    results['seg_id'] = test_set['seg_id']
    results[config.target_name] = model.predict(test_set[features])

    results.to_csv(config.path_to_results, index=False, float_format='%.5f')


if __name__ == '__main__':
    submit()

