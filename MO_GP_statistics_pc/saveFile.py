import csv
import hashlib
import json
import math
import os
import pickle
import re
import tempfile
from pathlib import Path

import numpy as np

def save_individual(randomSeeds, dataSetName,individuals):
    with open('./MO_GP_statistics_pc/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_' + dataSetName+'.pickle', 'wb') as file:
        pickle.dump(individuals, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    return

def save_each_gen_best_individual_meng(randomSeeds, dataSetName, best_ind_all_gen):
    individual_dict = {}

    for gen in range(len(best_ind_all_gen)):
        best_ind = best_ind_all_gen[gen]

        if len(best_ind)==2:
            sequencing = best_ind[0]
            routing = best_ind[1]
        else:
            sequencing = best_ind[0]

        individual = []
        sequencing_list = []
        for i in range(len(sequencing)):
            sequencing_list.append(sequencing[i].name)

        if len(best_ind) == 2:
            routing_list = []
            for i in range(len(routing)):
                routing_list.append(routing[i].name)

        individual.append(sequencing_list)
        if len(best_ind) == 2:
            individual.append(routing_list)

        individual_dict.__setitem__(gen, individual)

    with open('./MO_GP_statistics_pc/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_kmeans_individual_' + dataSetName + '.pkl', "wb") as fileName_individual:
        pickle.dump(individual_dict , fileName_individual)

    return

def save_top_inds_final_gen_meng(randomSeeds, pop_size, cross_rate, mute_rate, t_ration, delta,top_inds_fitness_final_gen):
    individual_dict = {}

    for gen in range(len(top_inds_fitness_final_gen)):
        best_ind = top_inds_fitness_final_gen[gen]

        if len(best_ind) == 2:
            sequencing = best_ind[0]
            routing = best_ind[1]
        else:
            sequencing = best_ind[0]

        individual = []
        sequencing_list = []
        for i in range(len(sequencing)):
            sequencing_list.append(sequencing[i].name)

        if len(best_ind) == 2:
            routing_list = []
            for i in range(len(routing)):
                routing_list.append(routing[i].name)

        individual.append(sequencing_list)
        if len(best_ind) == 2:
            individual.append(routing_list)

        individual_dict.__setitem__(gen, individual)

    with open('./MO_GP_statistics_pc/train/' + str(randomSeeds) + '_MOGPD_top_individuals_final_gen_statistics_pc_mut'+'_'+str(pop_size)+'_'+str(cross_rate)+'_'+str(mute_rate)+'_'+str(t_ration)+'_'+str(delta)+'_'+ '.pkl', "wb") as fileName_individual:
        pickle.dump(individual_dict , fileName_individual)

    return

def save_top_inds_final_gen(randomSeeds, top_inds_fitness_final_gen):
    individual_dict = {}

    for gen in range(len(top_inds_fitness_final_gen)):
        best_ind = top_inds_fitness_final_gen[gen]

        if len(best_ind) == 2:
            sequencing = best_ind[0]
            routing = best_ind[1]
        else:
            sequencing = best_ind[0]

        individual = []
        sequencing_list = []
        for i in range(len(sequencing)):
            sequencing_list.append(sequencing[i].name)

        if len(best_ind) == 2:
            routing_list = []
            for i in range(len(routing)):
                routing_list.append(routing[i].name)

        individual.append(sequencing_list)
        if len(best_ind) == 2:
            individual.append(routing_list)

        individual_dict.__setitem__(gen, individual)

    output_path = Path('./MO_GP_statistics_pc/train/' + str(randomSeeds) + '_MOGPD_top_individuals_final_gen_statistics_pc_mut'+ '.pkl')
    with output_path.open("wb") as fileName_individual:
        pickle.dump(individual_dict , fileName_individual)

    return output_path

def save_individual_to_txt(randomSeeds, dataSetName,individuals): 
    file = open('./MO_GP_statistics_pc/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_' + dataSetName+'.txt', 'w')
    file.write('Individual:\n')
    file.write('Tree 0:\n')
    file.write(str(individuals[0]) + '\n')
    if len(individuals) == 2:
        file.write('Tree 1:\n')
        file.write(str(individuals[1]) + '\n')

    file.close()
    return

def clear_individual_each_gen_to_txt(randomSeeds, dataSetName): 
    file = open('./MO_GP_statistics_pc/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_' + dataSetName+'_each_gen.txt', 'w') 
    file.write("Best individuals from each gen:\n")
    file.close()
    return

def save_individual_each_gen_to_txt(randomSeeds, dataSetName, individuals, gen): 
    file = open('./MO_GP_statistics_pc/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_' + dataSetName+'_each_gen.txt', 'a') 
    file.write('Individual:\n')
    file.write('Tree 0:\n')
    file.write(str(individuals[0]) + '\n')
    if len(individuals) == 2:
        file.write('Tree 1:\n')
        file.write(str(individuals[1]) + '\n')

    file.close()
    return

def save_top_inds_with_fitness_final_gen_to_txt(randomSeeds, dataSetName, individuals, fitnesses):
    file = open('./MO_GP_statistics_pc/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_' + dataSetName+'_top_inds_with_fitness_final_gen.txt', 'w')
    for i in range(len(individuals)):
        individual=individuals[i]
        file.write('Individual:' + str(i) + '\n')
        file.write('Tree 0:\n')
        file.write(str(individual[0]) + '\n')
        if len(individual) ==2:
            file.write('Tree 1:\n')
            file.write(str(individual[1]) + '\n')
        file.write('Fitness:\n')
        file.write(str(fitnesses[i]) + '\n')
        file.write('\n')

    file.close()
    return

def save_archive(randomSeeds, dataSetName,individuals):
    with open('./MO_GP_statistics_pc/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_archive' + dataSetName+'.pickle', 'wb') as file:
        pickle.dump(individuals, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    return

def save_pop(randomSeeds, dataSetName,individuals):
    with open('./MO_GP_statistics_pc/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_pop' + dataSetName+'.pickle', 'wb') as file:
        pickle.dump(individuals, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    return

def saveMinFitness(randomSeeds, dataSetName, min_fitness):
    fileName1= './MO_GP_statistics_pc/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds)+'_min_fitness' + dataSetName
    np.save(fileName1, min_fitness)
    return

def save_top_inds_fitness_final_gen(randomSeeds, dataSetName, min_fitness):
    fileName1= './MO_GP_statistics_pc/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds)+'_top_inds_fitness_final_gen' + dataSetName
    np.save(fileName1, min_fitness)
    return

def _atomic_replace(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise
    return path

def save_training_duration(randomSeeds, running_time, run_tag=None):
    tag = str(run_tag or "default")
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("._") or "default"
    output_dir = Path(__file__).resolve().parent / "train" / "training_durations"
    output_path = output_dir / f"{tag}_seed_{randomSeeds}_training_seconds.txt"
    return _atomic_replace(output_path, lambda handle: handle.write(f"{float(running_time):.17g}\n"))

def save_logbook_csv(randomSeeds, logbook, run_tag=None):
    tag = str(run_tag or "default")
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag).strip("._") or "default"
    output_dir = Path(__file__).resolve().parent / "train" / "training_logbooks"
    output_path = output_dir / f"{tag}_seed_{randomSeeds}_objectives.csv"
    columns = list(logbook.header or [])
    rows = list(logbook)
    def write(handle):
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return _atomic_replace(output_path, write)

def _artifact_path(path_value):
    return Path(path_value).resolve()

def _valid_objective_csv(path, expected_ngen=None):
    try:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            expected = ["gen", "nevals", "f0_avg", "f0_std", "f0_min", "f0_max", "f1_avg", "f1_std", "f1_min", "f1_max"]
            if reader.fieldnames != expected:
                return False
            rows = list(reader)
        if expected_ngen is not None:
            generations = [int(row["gen"]) for row in rows]
            if generations != list(range(int(expected_ngen) + 1)):
                return False
        return bool(rows)
    except (OSError, ValueError, TypeError, csv.Error, KeyError):
        return False

def _valid_tree(path):
    try:
        with Path(path).open("rb") as handle:
            value = pickle.load(handle)
        return isinstance(value, dict) and bool(value)
    except (OSError, EOFError, ValueError, TypeError, pickle.PickleError, AttributeError, ImportError):
        return False

def _valid_duration(path):
    try:
        value = float(Path(path).read_text(encoding="utf-8").strip())
        return math.isfinite(value) and value >= 0
    except (OSError, ValueError, TypeError):
        return False

def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def legacy_tree_path(randomSeeds):
    return Path(__file__).resolve().parent / "train" / f"{int(randomSeeds)}_MOGPD_top_individuals_final_gen_statistics_pc_mut.pkl"

def has_completed_pickle(randomSeeds):
    path = legacy_tree_path(randomSeeds)
    try:
        with path.open("rb") as handle:
            value = pickle.load(handle)
        return isinstance(value, dict) and bool(value)
    except (OSError, EOFError, ValueError, TypeError, pickle.PickleError, AttributeError, ImportError):
        return False

    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_tag or "default")).strip("._") or "default"
    marker = Path(__file__).resolve().parent / "train" / f"{tag}_seed_{randomSeeds}_complete.json"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            return False
        if payload.get("seed") != int(randomSeeds) or payload.get("run_tag") != tag:
            return False
        if payload.get("config_fingerprint") != str(config_fingerprint):
            return False
        paths = {key: _artifact_path(payload[key]) for key in ("tree", "duration", "objectives")}
        if not (_valid_tree(paths["tree"]) and _valid_duration(paths["duration"]) and _valid_objective_csv(paths["objectives"], expected_ngen)):
            return False
        artifacts = payload.get("artifacts", {})
        return all(artifacts.get(key, {}).get("sha256") == _sha256(path) for key, path in paths.items())
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return False

def save_completion_marker(randomSeeds, run_tag, config_fingerprint, tree_path, duration_path, objective_path):
    paths = [Path(tree_path), Path(duration_path), Path(objective_path)]
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        raise ValueError("required completion artifacts are missing or empty")
    try:
        if not (_valid_tree(paths[0]) and _valid_duration(paths[1]) and _valid_objective_csv(paths[2])):
            raise ValueError("required completion artifacts are invalid")
    except (OSError, ValueError, TypeError, pickle.PickleError, EOFError, AttributeError, ImportError) as exc:
        raise ValueError("required completion artifacts are invalid") from exc
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_tag or "default")).strip("._") or "default"
    marker = Path(__file__).resolve().parent / "train" / f"{tag}_seed_{randomSeeds}_complete.json"
    payload = {
        "schema_version": 2,
        "status": "complete",
        "seed": int(randomSeeds),
        "run_tag": tag,
        "config_fingerprint": str(config_fingerprint),
        "tree": str(paths[0]),
        "duration": str(paths[1]),
        "objectives": str(paths[2]),
        "artifacts": {
            "tree": {"sha256": _sha256(paths[0])},
            "duration": {"sha256": _sha256(paths[1])},
            "objectives": {"sha256": _sha256(paths[2])},
        },
    }
    return _atomic_replace(marker, lambda handle: json.dump(payload, handle, sort_keys=True, separators=(",", ":")))

def saveRunningTime(randomSeeds, dataSetName, running_time):
    fileName1= './MO_GP_statistics_pc/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds)+'_running_time' + dataSetName
    np.save(fileName1, running_time)
    return
