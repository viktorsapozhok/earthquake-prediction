# -*- coding: utf-8 -*-

"""Feature selection based on genetic algorithm
"""

import random
import logging

import numpy as np
import pandas as pd

from deap import creator, base, tools, algorithms
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, KFold

import config

logging.basicConfig(format='%(asctime)s | %(name)s | %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('ga')


class Chromosome(object):
    def __init__(self, genes, size):
        self.genes = self.generate(genes, size)

    def __repr__(self):
        return ' '.join(self.genes)

    def __get__(self, instance, owner):
        return self.genes

    def __set__(self, instance, value):
        self.genes = value

    def __getitem__(self, item):
        return self.genes[item]

    def __setitem__(self, key, value):
        self.genes[key] = value

    def __len__(self):
        return len(self.genes)

    @staticmethod
    def generate(genes, size):
        return random.sample(genes, size)


def init_individual(ind_class, genes=None, size=None):
    return ind_class(genes, size)


def evaluate(individual, model=None, train=None, n_splits=5, n_jobs=1):
    x = train[individual.genes]
    y = train['target']
    mae_folds = cross_val_score(model, x, y, cv=n_splits, scoring='neg_mean_absolute_error', n_jobs=n_jobs)
    mae = abs(mae_folds.mean())
    logger.info('%5.3f << %s' % (mae, individual))
    logger.info('')
    return mae,


def mutate(individual, genes=None, pb=0):
    # set the maximal amount of mutated genes
    n_mutated_max = max(1, int(len(individual) * pb))

    # generate the random amount of mutated genes
    n_mutated = random.randint(1, n_mutated_max)

    # randomly pick up genes which need to be mutated
    mutated_indexes = random.sample([index for index in range(len(individual.genes))], n_mutated)

    # mutate
    for index in mutated_indexes:
        individual[index] = random.choice(genes)

    return individual,


def get_data(path_to_file):
    data = pd.read_csv(path_to_file)
    data = data.replace([np.inf, -np.inf], np.nan)
    data.fillna(method='bfill', inplace=True)
    data.fillna(value=0, inplace=True)
    return data


def main():
    train = get_data(config.path_to_train)
    genes = [column for column in train.columns if column not in ['target', 'seg_id']]

    creator.create('FitnessMin', base.Fitness, weights=(-1,))
    creator.create('Individual', Chromosome, fitness=creator.FitnessMin)

    # register callbacks
    toolbox = base.Toolbox()
    toolbox.register('individual', init_individual, creator.Individual, genes=genes, size=15)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # raise population
    pop = toolbox.population(50)
    pop[0].genes = [
        'mfcc_15_avg', 'std_roll_mean_100', 'ffti_time_rev_asym_stat_10', 'mfcc_4_avg',
        'fftr_percentile_roll_std_80_window_10000', 'percentile_roll_std_20_window_1000',
        'ffti_exp_Moving_average_30000_mean', 'fftr_time_rev_asym_stat_100',
        'fftr_percentile_roll_std_30_window_100', 'ffti_count_big_50000_threshold_5',
        'percentile_roll_std_25_window_1000', 'percentile_roll_std_20_window_100',
        'percentile_roll_std_40_window_100', 'fftr_percentile_roll_std_1_window_50',
        'percentile_roll_std_40_window_1000'
    ]

    pop[1].genes = [
        'mfcc_15_avg', 'std_roll_mean_100', 'ffti_time_rev_asym_stat_10', 'mfcc_4_avg',
        'fftr_percentile_roll_std_80_window_10000', 'percentile_roll_std_20_window_1000',
        'ffti_exp_Moving_average_30000_mean', 'fftr_time_rev_asym_stat_100',
        'fftr_percentile_roll_std_30_window_100', 'percentile_roll_std_30_window_50',
        'fftr_num_peaks_100', 'ffti_mfcc_7_avg', 'ffti_classic_sta_lta3_mean',
        'fftr_percentile_roll_std_1_window_50', 'percentile_roll_std_40_window_1000'
    ]

    # set the model for evaluation of fitness function
    model = RandomForestRegressor(n_estimators=100, random_state=0)

    hof = tools.HallOfFame(5)

    # register fitness evaluator
    toolbox.register('evaluate', evaluate, model=model, train=train, n_splits=5, n_jobs=config.n_jobs)
    # crossover
    toolbox.register('mate', tools.cxTwoPoint)
    # mutation
    toolbox.register('mutate', mutate, genes=genes, pb=0.2)
    # register elitism operator
    toolbox.register('select', tools.selBest)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    try:
        algorithms.eaMuPlusLambda(
            pop, toolbox,
            mu=10, lambda_=30, cxpb=0.5, mutpb=0.5,
            ngen=50, stats=stats, halloffame=hof, verbose=True)
    except (Exception, KeyboardInterrupt):
        for individual in hof:
            logging.info('hof: %.3f << %s' % (individual.fitness.values[0], individual))

    for individual in hof:
        logging.info('hof: %.3f << %s' % (individual.fitness.values[0], individual))


if __name__ == '__main__':
    main()

