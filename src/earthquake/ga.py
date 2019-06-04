# -*- coding: utf-8 -*-

"""Feature selection based on genetic algorithm
"""

import random
import logging

import numpy as np
import pandas as pd

from deap import creator, base, tools, algorithms
from sklearn.model_selection import cross_val_score
from catboost import CatBoostRegressor

from operator import attrgetter

import config

logging.basicConfig(format='%(asctime)s | %(name)s | %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('ga')


class Chromosome(object):
    """
    Chromosome represents the list of genes, whereas each gene is the name of feature.
    Creating the chromosome we generate the random sample of features
    """
    def __init__(self, genes, size):
        """
        :param genes: list of all feature names
        :param size: number of genes in chromosome, i.e. number of features in the model
        """
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
    """Fitness value of the individual is introduced as the mean average error
    of the model with the set of features given by the individual (chromosome).

    :param individual: list of features (genes)
    :param model: estimator
    :param train: pandas dataframe contained all the features and target column
    :param n_splits: amount of splits in cross-validation
    :param n_jobs: amount of parallel jobs
    :return: mean average error calculated over cv-folds (tuple)
    """
    x = train[individual.genes]
    y = train['target']
    mae_folds = cross_val_score(model, x, y, cv=n_splits, scoring='neg_mean_absolute_error', n_jobs=n_jobs)
    mae = abs(mae_folds.mean())
    logger.info('%5.3f << %s' % (mae, individual))
    logger.info('')
    return mae,


def mutate(individual, genes=None, pb=0):
    """Custom mutation operator which is used instead of standard tools

    We define the maximal number of genes which can be mutated,
    then generate a random number of mutated genes (from 1 to max),
    and make a mutation.

    :param individual: list of features (genes)
    :param genes: list of all possible features
    :param pb: mutation parameter, 0 < pb < 1
    :return: mutated individual (tuple)
    """

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


def select_best(individuals, k, fit_attr="fitness"):
    """Custom select operator

    The only difference with standard 'selBest' method (select k best individuals)
    is that this method doesnt select two individuals with equal fitness value.
    It is done to prevent populations with many duplicate individuals
    """
    return sorted(set(individuals), key=attrgetter(fit_attr), reverse=True)[:k]


def get_data(path_to_file):
    """read data from .csv file and replace nans
    """
    data = pd.read_csv(path_to_file)
    data = data.replace([np.inf, -np.inf], np.nan)
    data.fillna(method='bfill', inplace=True)
    data.fillna(value=0, inplace=True)
    return data


def main():
    train = get_data(config.path_to_train)
    # full list of all possible features
    genes = [column for column in train.columns if column not in ['target', 'seg_id']]

    # set individual creator
    creator.create('FitnessMin', base.Fitness, weights=(-1,))
    creator.create('Individual', Chromosome, fitness=creator.FitnessMin)

    # register callbacks
    toolbox = base.Toolbox()
    toolbox.register('individual', init_individual, creator.Individual, genes=genes, size=config.n_features)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # raise population
    pop = toolbox.population(50)

    # set the model for evaluation of fitness function
    model = CatBoostRegressor(iterations=60, learning_rate=0.2, random_seed=0, verbose=False)

    # keep track of the best individuals
    hof = tools.HallOfFame(5)

    # register fitness evaluator
    toolbox.register('evaluate', evaluate, model=model, train=train, n_splits=5, n_jobs=config.n_jobs)
    # using standard crossover
    toolbox.register('mate', tools.cxTwoPoint)
    # replace mutation operator by custom method
    toolbox.register('mutate', mutate, genes=genes, pb=0.2)
    # register elitism operator
    toolbox.register('select', select_best)

    # set the statistics (displayed for each generation)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    # mu: the number of individuals to select for the next generation
    # lambda: the number of children to produce at each generation
    # cxpb: the probability that offspring is produced by crossover
    # mutpb: the probability that offspring is produced by mutation
    # ngen: the number of generations
    try:
        algorithms.eaMuPlusLambda(
            pop, toolbox,
            mu=10, lambda_=30, cxpb=0.2, mutpb=0.8,
            ngen=50, stats=stats, halloffame=hof, verbose=True)
    except (Exception, KeyboardInterrupt):
        for individual in hof:
            logging.info('hof: %.3f << %s' % (individual.fitness.values[0], individual))

    for individual in hof:
        logging.info('hof: %.3f << %s' % (individual.fitness.values[0], individual))


if __name__ == '__main__':
    main()

