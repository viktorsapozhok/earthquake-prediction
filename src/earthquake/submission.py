# -*- coding: utf-8 -*-

"""Make prediction and prepare results for submission
"""

import pandas as pd
from catboost import CatBoostRegressor

import config


def init_features():
    features = [
        'mfcc_15_avg', 'ffti_av_change_rate_roll_mean_500', 'ffti_time_rev_asym_stat_10', 'mfcc_4_avg',
        'fftr_percentile_roll_std_80_window_10000', 'percentile_roll_std_20_window_1000',
        'ffti_exp_Moving_average_30000_mean', 'fftr_time_rev_asym_stat_100',
        'fftr_percentile_roll_std_30_window_100', 'percentile_roll_std_30_window_50',
        'fftr_num_peaks_100', 'ffti_mfcc_7_avg', 'ffti_classic_sta_lta3_mean',
        'fftr_percentile_roll_std_1_window_50', 'percentile_roll_std_40_window_1000'
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

