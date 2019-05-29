# -*- coding: utf-8 -*-

"""Feature engineering using genetic algorithm
"""

import os
import random
import logging
from copy import deepcopy

import numpy as np
import pandas as pd

from deap import creator, base, tools, algorithms
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor, Pool

from src.earthquake import operators

logging.basicConfig(format='%(asctime)s | %(name)s | %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('ga')

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, '../..')))
PATH_TO_TRAIN = os.path.join(ROOT_DIR, 'data', 'train_int16float32.hdf')
TRAIN_SIZE = 629145480
SEGMENT_SIZE = 150000
LARGE_MAE = 3.999


class Chromosome(object):
    def __init__(self, slices=None, transforms=None, aggregations=None, n_transforms_max=0):
        self.genes = self.generate(slices, transforms, aggregations, n_transforms_max)

    def __repr__(self):
        return ' '.join([gene.get_name() for gene in self.genes])

    def set(self, genes):
        self.genes = genes

    def get_genes(self):
        return self.genes

    def get_gene(self, index):
        return self.genes[index]

    def get_aggregation_gene(self):
        return self.genes[-1]

    def set_gene(self, index, gene):
        self.genes[index] = gene

    def add_gene(self, index, gene):
        self.genes.insert(index, gene)

    def set_aggregation_gene(self, gene):
        self.genes[-1] = gene

    @staticmethod
    def generate(slices, transforms, aggregations, n_transforms_max):
        genes = []

        if transforms is None:
            return genes

        # generate first gene which must be the slice
        genes += [random.choice(slices)]

        # generate number of transformation genes
        n_transform_genes = random.randint(0, n_transforms_max)
        _transforms = transforms.copy()

        for i in range(n_transform_genes):
            # generate the index of new transformation gene
            j = random.randint(0, len(_transforms) - 1)

            # add new gene to individual
            genes += [_transforms[j]]

            # delete transformation from list if it cannot be used
            # multiple times (f.ex. detrend)
            if not _transforms[j].is_multiple():
                del _transforms[j]

            # exit loop if list is empty
            if len(_transforms) == 0:
                break

        # generate last gene which must be the aggregation
        genes += [random.choice(aggregations)]

        return genes


def init_individual(ind_class, slices=None, transforms=None, aggregations=None, n_transforms_max=5):
    ind = ind_class(slices=slices, transforms=transforms,
                    aggregations=aggregations, n_transforms_max=n_transforms_max)
    return ind


def create_set(individuals, seg_size, n_segments):
    x = []
    y = []

    gap_size = int((TRAIN_SIZE - (seg_size * n_segments)) / (n_segments - 1))
    start = 0

    for i in range(n_segments):
        # get segment from dataset
        #        with Segment(start=None, seg_size=seg_size) as segment:
        with Segment(start=start, seg_size=seg_size) as segment:
            signals = []

            # consecutively apply operators to signal
            for ind in individuals:
                signal = segment[0]

                for gene in ind.get_genes():
                    signal = gene.apply_soft(signal)

                signals += [signal]

            x += [signals] if len(individuals) > 1 else signals
            y += [segment[1]]

            del segment

        # increment start position
        start += seg_size + random.randint(1, gap_size)

    return x, y


def evaluate(individual, best=None, model=None, n_train_obs=None, n_test_obs=None, seg_size=None):
    logger.info('%5s << %s' % ('', individual))
    segmentation_size = (n_train_obs + n_test_obs) * seg_size

    # select random start position of the segmentation
    start = random.randint(0, TRAIN_SIZE - segmentation_size - 1)

    if len(best) == 0:
        individuals = [individual]
    else:
        individuals = [individual] + [ind for ind in best]

    # make train and test set
    x, y = create_set(individuals, seg_size, n_train_obs + n_test_obs)
    train_idx = random.sample(range(len(y)), n_train_obs)
    test_idx = [i for i in range(len(y)) if i not in train_idx]

    train_x = np.asarray([x[i] for i in train_idx])
    train_y = np.asarray([y[i] for i in train_idx])
    test_x = np.asarray([x[i] for i in test_idx])
    test_y = np.asarray([y[i] for i in test_idx])

    if len(best) == 0:
        train_x = np.reshape(np.repeat(train_x, 2), (-1, 2))
        test_x = np.reshape(np.repeat(test_x, 2), (-1, 2))

    # train the model
    try:
        model.fit(train_x, train_y)
        # make a prediction
        y_hat = model.predict(test_x)
        # calculate mean average error
        mae = np.mean(np.abs(test_y - y_hat))
        assert mae >= 0
    except (ValueError, AssertionError):
        mae = LARGE_MAE

    logger.info('%5.3f << %s' % (mae, individual))

    return mae,


def crossover(ind1, ind2):
    # amount of genes of both individuals
    n_genes_1 = len(ind1.get_genes())
    n_genes_2 = len(ind2.get_genes())

    # indexes of crossed genes
    index_1 = random.randint(0, n_genes_1 - 1)

    if index_1 == 0:
        crossed_gene = deepcopy(ind1.get_gene(0))
        ind1.set_gene(0, ind2.get_gene(0))
        ind2.set_gene(0, crossed_gene)

        return ind1, ind2

    # if one of the crossed genes is the aggregation gene then cross it
    # with the aggregation gene of the second individual
    if (index_1 == (n_genes_1 - 1)) or (n_genes_2 == 2):
        crossed_gene = deepcopy(ind1.get_aggregation_gene())
        ind1.set_aggregation_gene(ind2.get_aggregation_gene())
        ind2.set_aggregation_gene(crossed_gene)

        return ind1, ind2

    # cross transformation genes
    index_2 = random.randint(1, n_genes_2 - 2)
    crossed_gene = deepcopy(ind1.get_gene(index_1))
    ind1.set_gene(index_1, ind2.get_gene(index_2))
    ind2.set_gene(index_2, crossed_gene)

    return ind1, ind2


def mutate(individual, slices=None, transforms=None, aggregations=None, pb=0):
    if random.random() < pb:
        index = random.randint(1, len(individual.get_genes()) - 1)
        individual.add_gene(index, random.choice(transforms))

    for i, gene in enumerate(individual.get_genes()):
        if random.random() < pb:
            if gene.is_slice():
                _slices = [s for s in slices if s.get_index() != gene.get_index()]
                individual.set_gene(i, random.choice(_slices))
            elif gene.is_aggregation():
                # remove gene's aggregation from the list
                _aggregations = [a for a in aggregations if a.get_index() != gene.get_index()]
                # mutate gene
                individual.set_aggregation_gene(random.choice(_aggregations))
            else:
                # remove gene's transform from the list
                _transforms = [t for t in transforms if t.get_index() != gene.get_index()]
                # mutate gene
                individual.set_gene(i, random.choice(_transforms))

    return individual,


def main():
    aggregations = [operators.StDev(), operators.Min(), operators.Average(),
                    operators.Median(), operators.Skew(), operators.Kurtosis(),
                    operators.Argmin(), operators.Argmax(), operators.Max(), operators.MinMax(),
                    operators.Q1(), operators.Q5(), operators.Q10(), operators.Q25(),
                    operators.Q75(), operators.Q90(), operators.Q95(), operators.Q99(),
                    operators.NumPeaks(), operators.NumPeaks75(), operators.NumPeaks90(), operators.NumPeaks95(),
                    operators.WelchFreq(), operators.WelchFreqAvg(), operators.WelchDensityAvg()]

    transforms = [operators.Abs(), operators.Power2(), operators.MedFilt5(), operators.Scale(), operators.Log(),
                  operators.Roll1000(), operators.Roll5000(), operators.Roll10000(), operators.Roll50000()]

    slices = [operators.First1000(), operators.First5000(), operators.First10000(),
              operators.First50000(), operators.Last1000(), operators.Last5000(),
              operators.Last10000(), operators.Last50000(), operators.Raw()]

#    operators.Hilbert()

    creator.create('FitnessMin', base.Fitness, weights=(-1,))
    creator.create('Individual', Chromosome, fitness=creator.FitnessMin)

    # register callbacks
    toolbox = base.Toolbox()
    toolbox.register('individual', init_individual,
                     creator.Individual, slices=slices, transforms=transforms,
                     aggregations=aggregations, n_transforms_max=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)

    # best
    best = toolbox.population(1)
    best[0].set([operators.Last5000(), operators.Q95()])

    # raise population
    pop = toolbox.population(20)
    pop[0].set([operators.Raw(), operators.Q5()])
    pop[1].set([operators.Raw(), operators.Q95()])
    pop[2].set([operators.Raw(), operators.WelchDensityAvg()])
    pop[3].set([operators.Raw(), operators.WelchFreqAvg()])
    pop[4].set([operators.Raw(), operators.WelchFreq()])

    # set the model for evaluation of fitness function
    model = RandomForestRegressor(n_estimators=20)
#    model = CatBoostRegressor(random_seed=0, loss_function='MAE', verbose=False, learning_rate=0.1)

    hof = tools.HallOfFame(5)

    # register fitness evaluator
    toolbox.register('evaluate', evaluate, best=best, model=model, n_train_obs=3000, n_test_obs=700, seg_size=150000)
    # crossover
    toolbox.register('mate', crossover)
    # mutation
    toolbox.register('mutate', mutate, slices=slices, transforms=transforms, aggregations=aggregations, pb=0.2)
    # register elitism operator
#    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('select', tools.selBest)

    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

#    algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=10, stats=stats, halloffame=hof, verbose=True)
    algorithms.eaMuPlusLambda(pop, toolbox,
                              mu=6, lambda_=10, cxpb=0.5, mutpb=0.5, ngen=5,
                              stats=stats, halloffame=hof, verbose=True)

    for ind in hof:
        logging.info('hof: %.3f << %s' % (ind.fitness.values[0], ind))


if __name__ == '__main__':
    main()

