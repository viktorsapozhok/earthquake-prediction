# -*- coding: utf-8 -*-

"""
notebooks utils (quick and dirty)
methods used in notebooks for exploration analysis
"""

import os
import logging

import tqdm

import numpy as np
import pandas as pd
from datetime import timedelta

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold
import rfpimp
from catboost import CatBoostRegressor, Pool

logging.basicConfig(format='%(asctime)s | %(name)s | %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('INTM')


def print_features_imp(x, y, logger, **kwargs):
    permutation = kwargs.get('permutation', False)
    min_imp = kwargs.get('min_imp', 0.05)

    model = RandomForestRegressor(n_estimators=50)
    model.fit(x, y)

    logger.info('%5s | %s' % ('Imp', 'Feature'))
    best_features = []

    # calculate features importance
    if permutation:
        # using permutations to improve mean decrease impurity mechanism
        feat_imp = rfpimp.importances(model, x, y)
        # print the results
        for index, row in feat_imp.iterrows():
            logger.info('%4.0f%% | %s' % (row['Importance'] * 100, index))

            if row['Importance'] > min_imp:
                best_features += [index]
    else:
        feat_imp = model.feature_importances_
        # sort features by importance
        feat_indexes = np.argsort(feat_imp)
        # print the results
        for id in feat_indexes[::-1]:
            logger.info('%4.0f%% | %s' % (feat_imp[id] * 100, x.columns[id]))

            if feat_imp[id] > min_imp:
                best_features += [x.columns[id]]

    return best_features


def tune_random_forest(X, y, logger, **kwargs):
    mdl_type = kwargs.get('mdl_type', 'regression')
    n_splits = kwargs.get('n_splits', 4)
    n_estimators = kwargs.get('n_estimators', [10])
    max_features = kwargs.get('max_features', ['auto'])
    leaf_size = kwargs.get('leaf_size', [1])
    criterion = kwargs.get('criterion', 'mse')

    if mdl_type == 'regression':
        alg = 'RandomForestRegressor'
    elif mdl_type == 'classifier':
        alg = 'RandomForestClassifier'
    else:
        alg = ''

    if max_features == 'all':
        max_features = range(1, 1 + len(X.columns))

    logger.info('%7s | %7s | %7s | %7s | %7s' % ('n_est', 'n_feats', 'n_leafs', 'MSE', 'MAE'))

    kfold = KFold(n_splits=n_splits, shuffle=False)
#    tscv = TimeSeriesSplit(n_splits=n_splits)

    for n_est in n_estimators:
        for max_feat in max_features:
            for ls in leaf_size:
                model = globals()[alg](
                    n_estimators=n_est,
                    max_features=max_feat,
                    min_samples_leaf=ls,
                    criterion=criterion)

#                mae = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
#                mse = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
                mae = cross_val_score(model, X, y, cv=n_splits, scoring='neg_mean_absolute_error')
#                mse = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')

                if max_feat == 'auto':
#                    logger.info('%7.0f | %7s | %7.0f | %7.2f | %7.2f' % (n_est, 'auto', ls, -mse.mean(), -mae.mean()))
                    logger.info('%7.0f | %7s | %7.0f | %7.2f | %7.2f' % (n_est, 'auto', ls, 0, -mae.mean()))
                else:
#                    logger.info('%7.0f | %7.0f | %7.0f | %7.2f | %7.2f' % (n_est, max_feat, ls, -mse.mean(), -mae.mean()))
                    logger.info('%7.0f | %7.0f | %7.0f | %7.2f | %7.2f | ' % (n_est, max_feat, ls, 0, -mae.mean()))
                    logger.info('scores: {}'.format(abs(mae)))


def tune_catboost(X, y, **kwargs):
    n_splits = kwargs.get('n_splits', 9)
    iterations = kwargs.get('iterations', [10])
    depth = kwargs.get('tree_depth', [4])
    random_strength = kwargs.get('random_strength', [1])
    learning_rate = kwargs.get('learning_rate', [0.03])
    loss_function = kwargs.get('loss_function', ['RMSE'])
    l2_leaf_reg = kwargs.get('l2_leaf_reg', [3])
    bagging_temperature = kwargs.get('bagging_temperature', [1])

    logger.info('%4s | %5s | %4s | %4s | %4s | %4s | %5s | %7s' %
                ('iter', 'depth', 'rand', 'rate', 'L2', 'temp', 'LF', 'MAE'))

    tscv = TimeSeriesSplit(n_splits=n_splits)

    for iter in iterations:
        for d in depth:
            for rs in random_strength:
                for r in learning_rate:
                    for lf in loss_function:
                        for l2 in l2_leaf_reg:
                            for bt in bagging_temperature:
                                model = CatBoostRegressor(
                                    iterations=iter,
                                    depth=d,
                                    random_strength=rs,
                                    learning_rate=r,
                                    l2_leaf_reg=l2,
                                    loss_function=lf,
                                    bagging_temperature=bt,
                                    verbose=False)

                                mae = cross_val_score(model, X, y, cv=n_splits, scoring='neg_mean_absolute_error')

                                logger.info('%4.0f | %5.0f | %4.1f | %4.2f | %4.1f | %4.1f | %5s | %7.2f' %
                                            (iter, d, rs, r, l2, bt, lf, abs(mae.mean())))

                            
def catboost_best_iter(model, logger, train_x, train_y, eval_x, eval_y):
    eval_set = Pool(eval_x, eval_y)
    model.fit(train_x, train_y, eval_set=eval_set, verbose=False)
    logger.info('best iteration: %.0f' % model.get_best_iteration())                            


def run_model(model, data, features, target, n_train_obs, n_fore_obs, n_fore_days):
    forecast_date_end = data.index.max().replace(hour=0)
    forecast_date_start = forecast_date_end + timedelta(days=-n_fore_days + 1)
    forecast_dates = utils.date_list(forecast_date_start, forecast_date_end)

    results = pd.DataFrame()

    for d in tqdm.tqdm(forecast_dates):
        train = data.query('date < @d').tail(n_train_obs)
        test = data.query('date >= @d').head(n_fore_obs)

        model.fit(train[features].values, train[target].values)

        res_iter = pd.DataFrame()
        res_iter['y'] = test[target]
        res_iter['y_hat'] = model.predict(test[features].values)

        results = results.append(res_iter)

    mse = np.mean((results['y'].values - results['y_hat'].values) ** 2)
    mae = np.mean(np.abs(results['y'].values - results['y_hat'].values))

    logger.info('MSE: %.2f, MAE: %.2f' % (mse, mae))

    return results


def calc_interval_accuracy(y, y_left, y_right, **kwargs):
    th = kwargs.get('threshold', 0)

    n_all = len(y)
    n_obs = sum([1 * ((y[i] >= y_left[i] - th) & (y[i] <= y_right[i] + th)) for i in range(n_all)])

    return n_obs / n_all


def eval_mdl_errors(data, logger, target, model, n_days):
    forecast_date_start = data.index.max().replace(hour=0)
    forecast_date_start += timedelta(days=-n_days + 1)
    test = data.query('date >= @forecast_date_start')
    mse = np.nanmean((test[target].values - test[model].values) ** 2)
    mae = np.nanmean(np.abs(test[target].values - test[model].values))
    logger.info('MSE: %.0f, MAE: %.2f' % (mse, mae))


def eval_results(res, y, y_hat, y_lower, y_upper, logger, threshold=5):
    target = res[y].values
    pred = res[y_hat].values

    mse = np.mean((target - pred) ** 2)
    mae = np.mean(np.abs(target - pred))

    dir = 1 * (target > 0.1) - 1 * (target < -0.1)
    dir_hat = 1 * (pred > 0.1) - 1 * (pred < -0.1)

    dir_acc = sum(dir_hat == dir) / len(dir)

    if (y_lower is not None) and (y_upper is not None):
        int_lower = res[y_lower].values - threshold
        int_upper = res[y_upper].values + threshold
        int_acc = eval_utils.ci_coverage(target, int_lower, int_upper, threshold=0)
        int_width = np.mean(np.abs(int_lower - int_upper))

    logger.info(' dir acc : %.0f%%' % (100 * dir_acc))
    logger.info('     MSE : %.0f' % mse)
    logger.info('     MAE : %.2f' % mae)

    if (y_lower is not None) and (y_upper is not None):
        logger.info('  CI acc : %.0f%%' % int_acc)
        logger.info('CI width : %.0f' % int_width)
