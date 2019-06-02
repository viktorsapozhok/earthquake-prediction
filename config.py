# -*- coding: utf-8 -*-

"""Configuration parameters
"""

import os

root_dir = os.path.dirname(os.path.abspath(__file__))

# path to .hdf file with initial training set
path_to_hdf = os.path.join(root_dir, 'data', 'train.hdf')

# path to created training set with all features
path_to_train = os.path.join(root_dir, 'data', 'train_4100.csv')

# path to .hdf file with testing set
path_to_test = os.path.join(root_dir, 'data', 'test_int16.hdf')

# path to prediction results
path_to_results = os.path.join(root_dir, 'data', 'results.csv')

# path to created test set with
path_to_test_set = os.path.join(root_dir, 'data', 'test.csv')

# signal column name
signal_name = 'acoustic_data'

# target column name
target_name = 'time_to_failure'

# amount of rows in initial training set
n_rows_all = 629145480

# amount of segments
n_segments = 4100

# amount of rows in segment
n_rows_seg = 150000

# amounts of observations requested in slices
slice_counts = [1000, 5000, 10000, 25000, 50000]

# list of quantiles used in feature engineering
quantiles = [0.01, 0.05, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999]

# list of quantiles used in peak detection
peak_quantiles = [0.95, 0.99, 0.999]

# distance parameter used in peak detection
peak_distance = 1000

# amount of parallel jobs
n_jobs = 4
