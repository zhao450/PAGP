import  pickle
import numpy as np
import sys

def load_individual_from_gen(randomSeeds, dataSetName):

    with open(sys.path[0] + '/MO_GP_statistics_pc/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_kmeans_individual_' + dataSetName + '.pkl',
            "rb") as fileName_individual:
        dict = pickle.load(fileName_individual)

    return dict

def load_top_inds_from_final_gen(randomSeeds):
    with open(sys.path[0] + '/MO_GP_statistics_pc/train/' + '/' + str(randomSeeds) + '_MOGPD_top_individuals_final_gen_statistics_pc_mut' + '.pkl',
            "rb") as fileName_individual:
        dict = pickle.load(fileName_individual)

    return dict

def load_training_time(randomSeeds, dataSetName):
    folder = './MO_GP_statistics_pc/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_running_time' + dataSetName + '.npy'
    training_time = np.load(folder)

    return training_time

def load_min_fitness(randomSeeds, dataSetName):
    folder = './MO_GP_statistics_pc/train/scenario_' + str(dataSetName) + '/' +  str(
        randomSeeds) + '_min_fitness' + dataSetName + '.npy'
    min_fitness = np.load(folder)

    return min_fitness
