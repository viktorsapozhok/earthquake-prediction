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

# path to created test set
path_to_test_set = os.path.join(root_dir, 'data', 'test.csv')

# signal column name
signal_name = 'acoustic_data'

# target column name
target_name = 'time_to_failure'

# amount of rows in initial training set
n_rows_all = 629145480
#n_rows_all = 1500000

# amount of segments
n_segments = 4100

# amount of rows in segment
n_rows_seg = 150000

# amount of parallel jobs
n_jobs = 8


