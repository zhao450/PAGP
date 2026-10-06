import math
import random

import numpy as np

from deap import tools

def sel_random(individuals, buckets, k):
    chosen_inds = []
    chosen_buckets = []
    for i in range(k):
        r = random.randrange(len(individuals))
        chosen_inds.append(individuals[r])
        chosen_buckets.append(buckets[r])

    return chosen_inds, chosen_buckets

def sel_least_complex(individuals, complexity_func):
    if len(individuals) == 1:
        return individuals[0]
    else:
        lowest_complexity = math.inf
        for ind in individuals:
            complexity = complexity_func(ind)
            if complexity < lowest_complexity:
                lowest_complexity = complexity
                least_complex = ind
        return least_complex

def selRandom(individuals, k):
    return [random.choice(individuals) for i in range(k)]

def selTournament(individuals, k, tournsize):
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        aspirants_fit = [np.sum(ind.fitness.values) for ind in aspirants]
        best_index = np.argmin(aspirants_fit)
        chosen.append(aspirants[best_index])
    return chosen

def selElitistAndTournament(individuals, k, tournsize, elitism):
    return selTournament(individuals, k, tournsize)
