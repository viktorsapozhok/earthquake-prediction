# -*- coding: utf-8 -*-

"""Configuration parameters
"""

import logging
import os

root_dir = os.path.dirname(os.path.abspath(__file__))

# path to raw training set
path_to_train_store = os.path.join(root_dir, 'data', 'train.hdf')

# path to training set with created features
path_to_train = os.path.join(root_dir, 'data', 'train.csv')

# path to raw testing set
path_to_test_store = os.path.join(root_dir, 'data', 'test.hdf')

# path to test set with created features
path_to_test = os.path.join(root_dir, 'data', 'test.csv')

# path to prediction results
path_to_results = os.path.join(root_dir, 'data', 'results.csv')

# amount of rows in raw training set
n_rows_all = 629145480

# amount of rows in segment
segment_size = 150000

# amount of parallel jobs
n_jobs = 8

# amount of features in the model
n_features = 15


def setup_logger():
    """Configure logger
    """
    logging.basicConfig(
        format='%(asctime)s | %(name)s | %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger('earthquake')

    return logger
