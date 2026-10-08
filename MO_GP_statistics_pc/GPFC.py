import simpy
from deap import base
from deap import creator
from deap import gp
from MO_GP_statistics_pc import ea_simple_elitism
from MO_GP_statistics_pc.ParallelToolbox import ParallelToolbox
from MO_GP_statistics_pc.selection import *
from MO_GP_statistics_pc.multi_tree import init_primitives_general, init_toolbox_dual

import sys
from MO_GP_statistics_pc import saveFile
import time
import random
import multiprocessing
import copy
import traceback

import numpy as np
import jobshop
import jobshop_pc

def connectedness(cluster):
    print(cluster)

def init_stats():
    fit0 = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    fit1 = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    for s in (fit0, fit1):
        s.register("avg", np.mean)
        s.register("std", np.std)
        s.register("min", np.min)
        s.register("max", np.max)
    mstats = tools.MultiStatistics(f0=fit0, f1=fit1)
    return mstats
def evaluate(individual, toolbox, seed, simulation_config=None, train_replications=None, seed_step=None):

    simulation_config = simulation_config or {}
    repetitions = train_replications or ins_each_gen
    seed_step = seed_step or 1000
    fitness_list = []
    if len(individual) == 2:

        fmax,tmax,wfmax,wtmax = jobshop.main(individual[0], individual[1],seed=seed, ifPrint=False,ifTest=False, simulation_config=simulation_config)

        fitness_list.append([fmax,wtmax])
        fitness = fmax
        wttdmax=wtmax
        for i in range(repetitions-1):
            seed = seed + seed_step
            fmax,tmax,wfmax,wtmax = jobshop.main(individual[0], individual[1],seed=seed,ifPrint=False,ifTest=False, simulation_config=simulation_config)
            fitness_list.append([fmax,wtmax])
            fitness = fitness + fmax
            wttdmax=wttdmax+wtmax
        fitness = fitness/repetitions
        wttdmax=wttdmax/repetitions
        scores = [fitness,wttdmax]

    return scores,fitness_list

def evaluate_one_rep(individual, seed):
    fmax,tmax,wfmax,wtmax = jobshop.main(individual[0], individual[1], seed=seed, ifPrint=False, ifTest=False)
    return fmax, wtmax

def evaluate_30_rep(individual, seed, simulation_config=None, final_replications=30, seed_step=1000):

    simulation_config = simulation_config or {}
    instance = final_replications
    fitness_list = []
    if len(individual) == 2:

        fmax,tmax,wfmax,wtmax = jobshop.main(individual[0], individual[1],seed=seed, ifPrint=False,ifTest=False, simulation_config=simulation_config)

        fitness_list.append([fmax,wtmax])
        fitness = fmax
        wttdmax=wtmax
        for i in range(instance-1):
            seed = seed + seed_step
            fmax,tmax,wfmax,wtmax = jobshop.main(individual[0], individual[1],seed=seed,ifPrint=False,ifTest=False, simulation_config=simulation_config)
            fitness_list.append([fmax,wtmax])
            fitness = fitness + fmax
            wttdmax=wttdmax+wtmax
        fitness = fitness/instance
        wttdmax=wttdmax/instance
        scores = [fitness,wttdmax]

    return scores,fitness_list

def eval_wrapper(individual, seed=None, *args, **kwargs):
    if seed is None:
        seed = rd.get('seed', None)
    try:
        tb = worker_toolbox
    except NameError:
        tb = rd.get('toolbox', None)
    return evaluate(
        individual,
        toolbox=tb,
        seed=seed,
        simulation_config=rd.get('simulation_config', {}),
        train_replications=rd.get('train_replications', ins_each_gen),
        seed_step=rd.get('seed_step', 1000),
    )

def eval_wrapper_30(individual, seed=None, *args, **kwargs):
    if seed is None:
        seed = rd.get('seed', None)
    try:
        tb = worker_toolbox
    except NameError:
        tb = rd.get('toolbox', None) 
    return evaluate_30_rep(
        individual,
        seed=seed,
        simulation_config=rd.get('simulation_config', {}),
        final_replications=rd.get('final_replications', 30),
        seed_step=rd.get('seed_step', 1000),
    )

def init_data(rundata):

    global rd
    rd = rundata

    try:

        if not hasattr(creator, "FitnessMin"):
            weights = (-1., -1.)
            creator.create("FitnessMin", base.Fitness, weights=weights)
        toolbox = base.Toolbox()

        pset_rou = gp.PrimitiveSet("ROU", 0, prefix="f")
        pset_seq = gp.PrimitiveSet("SEQ", 0, prefix="f")
        init_primitives_general(pset_rou)
        init_primitives_general(pset_seq)
        init_toolbox_dual(toolbox, pset_rou, pset_seq)

        toolbox.register("evaluate", eval_wrapper)
        toolbox.register("evaluate_one_rep", evaluate_one_rep)
        toolbox.register("evaluate_30_rep", eval_wrapper_30)

        global worker_toolbox
        worker_toolbox = toolbox
    except Exception as e:

        traceback.print_exc()
        raise

def init_ref_vectors_2obj(K):
    lambdas = []
    for i in range(K):
        lam1 = i / (K - 1)
        lambdas.append(np.array([lam1, 1.0 - lam1], dtype=float))
    return np.array(lambdas)  
def build_neighbors(lambdas, T):
    K = len(lambdas)
    dmat = np.linalg.norm(lambdas[:, None, :] - lambdas[None, :, :], axis=2)
    neigh = []
    for k in range(K):
        idx = np.argsort(dmat[k])[:T]  
        neigh.append(list(idx))
    return neigh 

def GPFC_main(seed,pop_size, cross_rate, mute_rate, t_ration, delta, simulation_config=None, algorithm_config=None, return_logbook=False):
    global rd
    simulation_config = simulation_config or {}
    algorithm_config = algorithm_config or {}
    rd['use_niching'] = use_niching
    rd['use_kmeans']= use_kmeans
    rd['seed'] = seed
    num_features = 0
    pset = gp.PrimitiveSet("MAIN", num_features, prefix="f")
    pset.context["array"] = np.array

    weights = (-1.,-1.)
    creator.create("FitnessMin", base.Fitness, weights=weights)

    manager = multiprocessing.Manager()
    rundata = manager.dict({
        'seed': seed,
        'use_niching': use_niching,
        'use_kmeans': use_kmeans,
        'only_sequencing_rule': only_sequencing_rule,
        'simulation_config': simulation_config,
        'train_replications': algorithm_config.get('train_replications', ins_each_gen),
        'seed_step': algorithm_config.get('seed_replication_step', 1000),
        'final_replications': algorithm_config.get('final_eval_replications', 30),
    })

    num_cores = multiprocessing.cpu_count()
    print(f"✅ 使用 {num_cores} 个CPU核心并行评估")
    pool = multiprocessing.Pool(
        processes=algorithm_config.get('pool_processes', 8),
        initializer=init_data,
        initargs=(rundata,)
    )
    toolbox=base.Toolbox()
    pset_rou = gp.PrimitiveSet("ROU", 0, prefix="f")
    pset_seq = gp.PrimitiveSet("SEQ", 0, prefix="f")
    init_primitives_general(pset_rou)
    init_primitives_general(pset_seq)
    init_toolbox_dual(toolbox, pset_rou, pset_seq)

    toolbox.register("evaluate", eval_wrapper)
    toolbox.register("evaluate_one_rep", evaluate_one_rep)
    toolbox.register("evaluate_30_rep", eval_wrapper_30)
    toolbox.register("select", selElitistAndTournament, tournsize=TOURNAMENT_SIZE, elitism=ELITISM)

    toolbox.register("map", pool.map)

    rd_local = rundata.copy()
    rd_local['toolbox'] = toolbox
    rd_local['seed'] = seed
    rd = rundata
    int_pop_size = int(pop_size)
    pop = toolbox.population(n=int_pop_size)
    stats = init_stats()
    hof = tools.HallOfFame(1)
    seedRotate = True

    lambdas = init_ref_vectors_2obj(int_pop_size)
    T = t_ration * int_pop_size

    neighbors = build_neighbors(lambdas, int(T))

    if_statistics=True
    if_pc=True
    if_semantic_mutation=True
    if_rebuild_update=True
    if_pc_penalty=True
    ea_ngen = int(algorithm_config.get('ngen', NGEN))
    ea_reppb = algorithm_config.get('reproduction_rate', REPPB)
    ea_nr = int(algorithm_config.get('nr', NR))
    ea_elitism = int(algorithm_config.get('elitism', ELITISM))
    pop, logbook, min_fitness, best_ind_all_gen, top_inds_fitness_final_gen, top_inds_final_gen = ea_simple_elitism.eaSimple(pop, toolbox, cross_rate, mute_rate, ea_reppb ,delta, ea_nr, ea_elitism, ea_ngen, seedRotate, rd, int_pop_size,lambdas, neighbors,if_statistics,if_pc,if_semantic_mutation,if_rebuild_update,if_pc_penalty,g_func='tch', stats=stats, halloffame=hof, verbose=True, seed =seed)

    pool.close()
    pool.join()

    best = hof[0]
    if return_logbook:
        return min_fitness,best, logbook, best_ind_all_gen, top_inds_fitness_final_gen, top_inds_final_gen
    return min_fitness,best, best_ind_all_gen, top_inds_fitness_final_gen, top_inds_final_gen

POP_SIZE =10
NGEN = 50
CXPB = 0.8
MUTPB = 0.15
REPPB = 0.05
ELITISM = 5
TOURNAMENT_SIZE = 4
MAX_HEIGHT = 8

DELTA= 0.9
NR=2
only_sequencing_rule = False
rd = {}
use_niching = False
use_kmeans=False

ins_each_gen = 2
def main(seed,pop_size, cross_rate, mute_rate, t_ration, delta, simulation_config=None, algorithm_config=None, run_tag=None, config_fingerprint=None):

    random.seed(int(seed))
    np.random.seed(int(seed))
    simulation_config = simulation_config or {}
    algorithm_config = algorithm_config or {}
    run_tag = run_tag or algorithm_config.get("run_tag")

    start = time.time()
    min_fitness,p_one,logbook,best_ind_all_gen, top_inds_fitness_final_gen, top_inds_final_gen = GPFC_main(
        seed,pop_size, cross_rate, mute_rate, t_ration, delta,
        simulation_config=simulation_config,
        algorithm_config=algorithm_config,
        return_logbook=True)
    end = time.time()
    running_time = end - start
    tree_path = saveFile.save_top_inds_final_gen(seed, top_inds_final_gen)
    duration_path = saveFile.save_training_duration(seed, running_time, run_tag=run_tag)
    objective_path = saveFile.save_logbook_csv(seed, logbook, run_tag=run_tag)
    if config_fingerprint is not None:
        saveFile.save_completion_marker(seed, run_tag, config_fingerprint, tree_path, duration_path, objective_path)
    print(min_fitness)
    print("Training time: " + str(running_time))
    print('Training end!')
    return running_time
