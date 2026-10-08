import hashlib
import json
import multiprocessing as mp
from pathlib import Path

TRAIN_SEEDS = list(range(30))

POP_SIZE = 160
NGEN = 50
CROSS_RATE = 0.95
MUTATION_RATE = 0.25
REPRODUCTION_RATE = 0.05
NEIGHBORHOOD_RATIO = 0.20
DELTA = 0.70
ELITISM = 5
NR = 2
TRAIN_REPLICATIONS = 2
FINAL_EVAL_REPLICATIONS = 30
SEED_REPLICATION_STEP = 1000
POOL_PROCESSES = 4

NUM_MACHINES = 5
TOTAL_JOBS = 124
WARMUP_JOBS = 16
OPS_PER_JOB = 10
WORKLOAD_MIN = 100
WORKLOAD_MAX = 1000
RATE_MIN = 10
RATE_MAX = 15
DUE_FACTOR = 1.5
UTIL_LEVEL = 0.85

INTERARRIVAL_MEAN = None
ARRIVAL_RATE_OVERRIDE = None

CANDIDATE_MIN = 2
CANDIDATE_MAX = None
WEIGHT_VALUES = (1, 2, 4)
WEIGHT_PROBS = (0.2, 0.6, 0.2)
POLL_INTERVAL = 0.1
SIMULATION_STOP_MODE = "all_jobs_completed"
PC_REFERENCE_SEED = 12345
RUN_TAG = "small_train"
def config_fingerprint(simulation_config, algorithm_config):
    payload = {"simulation": simulation_config, "algorithm": algorithm_config}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=list).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def unique_seeds(seeds):
    return list(dict.fromkeys(int(seed) for seed in seeds))

def build_simulation_config():
    return {
        "num_machines": NUM_MACHINES,
        "total_jobs": TOTAL_JOBS,
        "warmup_jobs": WARMUP_JOBS,
        "ops_per_job": OPS_PER_JOB,
        "workload_min": WORKLOAD_MIN,
        "workload_max": WORKLOAD_MAX,
        "rate_min": RATE_MIN,
        "rate_max": RATE_MAX,
        "due_factor": DUE_FACTOR,
        "util_level": UTIL_LEVEL,
        "arrival_rate_override": ARRIVAL_RATE_OVERRIDE,
        "interarrival_mean": INTERARRIVAL_MEAN,
        "candidate_min": CANDIDATE_MIN,
        "candidate_max": CANDIDATE_MAX,
        "weight_values": WEIGHT_VALUES,
        "weight_probs": WEIGHT_PROBS,
        "poll_interval": POLL_INTERVAL,
        "simulation_stop_mode": SIMULATION_STOP_MODE,
        "pc_reference_seed": PC_REFERENCE_SEED,
    }

def build_algorithm_config():
    return {
        "pop_size": POP_SIZE,
        "ngen": NGEN,
        "cross_rate": CROSS_RATE,
        "mutation_rate": MUTATION_RATE,
        "reproduction_rate": REPRODUCTION_RATE,
        "neighborhood_ratio": NEIGHBORHOOD_RATIO,
        "delta": DELTA,
        "elitism": ELITISM,
        "nr": NR,
        "train_replications": TRAIN_REPLICATIONS,
        "final_eval_replications": FINAL_EVAL_REPLICATIONS,
        "seed_replication_step": SEED_REPLICATION_STEP,
        "pool_processes": POOL_PROCESSES,
        "run_tag": RUN_TAG,
    }

if __name__ == "__main__":
    mp.freeze_support()
    from MO_GP_statistics_pc import GPFC

    simulation_config = build_simulation_config()
    algorithm_config = build_algorithm_config()
    fingerprint = config_fingerprint(simulation_config, algorithm_config)
    seeds = unique_seeds(TRAIN_SEEDS)
    for seed in seeds:
        if GPFC.saveFile.has_completed_pickle(seed):
            print(f"Skipping seed {seed}: final population pickle already exists")
            continue
        GPFC.main(
            seed=seed,
            pop_size=algorithm_config["pop_size"],
            cross_rate=algorithm_config["cross_rate"],
            mute_rate=algorithm_config["mutation_rate"],
            t_ration=algorithm_config["neighborhood_ratio"],
            delta=algorithm_config["delta"],
            simulation_config=simulation_config,
            algorithm_config=algorithm_config,
            run_tag=algorithm_config["run_tag"],
            config_fingerprint=fingerprint,
        )
