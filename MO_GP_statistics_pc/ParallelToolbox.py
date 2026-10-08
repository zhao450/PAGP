import pickle

from deap import base
from multiprocessing import cpu_count, Pool

class ParallelToolbox(base.Toolbox):

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['map']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def multiProcess(self, evaluate, invalid_ind):
        cores = cpu_count()

        pickle.dumps(invalid_ind)
        pickle.dumps(evaluate)
        fitnesses = Pool().map(evaluate, invalid_ind)
        return fitnesses
