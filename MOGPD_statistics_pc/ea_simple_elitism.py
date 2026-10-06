import random

import numpy as np,math
import hashlib
from deap import tools

from MOGPD_statistics_pc import saveFile
from MOGPD_statistics_pc.selection import selElitistAndTournament
from MOGPD_statistics_pc.calculate_pc import computer_PC
import math
from math import dist, log10
from scipy.stats import norm
from scipy.stats import t as t_dist
import jobshop
import jobshop_pc

def eval_one_rep_task(args):
    ind, rep_seed = args
    """
    给 Pool 用的任务：输入 (ind, seed) -> 输出 (f0,f1)
    必须放在模块顶层，才能被 multiprocessing pickle。
    """

    fmax, tmax, wfmax, wtmax = jobshop.main(ind[0], ind[1], seed=int(rep_seed), ifPrint=False, ifTest=False)
    return float(fmax), float(wtmax)

def _eval_one_rep_cached_parallel(toolbox, tasks, cache: dict):
    need = []
    need_keys = []
    for ind, s in tasks:
        key = (_ind_key(ind), int(s))
        if key not in cache:
            need.append((ind, int(s)))
            need_keys.append(key)

    if need:

        res = toolbox.map(eval_one_rep_task, need)
        for k, (f0, f1) in zip(need_keys, res):
            cache[k] = (float(f0), float(f1))

    return [cache[(_ind_key(ind), int(s))] for ind, s in tasks]

def decide_replace_parallel(child, old, lambdaj, z, scale, rep_seeds, toolbox, cache,
                            R_min=2, R_max=8, alpha=0.05, tau=0.0,
                            penalty_weight=None, penalty_func=None):
    seeds = list(map(int, rep_seeds[:R_max]))

    tasks = [(child, s) for s in seeds] + [(old, s) for s in seeds]
    results = _eval_one_rep_cached_parallel(toolbox, tasks, cache)
    child_res = results[:R_max]
    old_res   = results[R_max:]

    diffs = []
    for r in range(1, R_max + 1):
        fc0, fc1 = child_res[r-1]
        fo0, fo1 = old_res[r-1]

        gc = g_tchebycheff_abs((fc0, fc1), lambdaj, z, scale)
        go = g_tchebycheff_abs((fo0, fo1), lambdaj, z, scale)

        if penalty_weight is not None and penalty_func is not None:
            gc += float(penalty_weight) * float(penalty_func(child))
            go += float(penalty_weight) * float(penalty_func(old))

        d = gc - go
        diffs.append(d)

        if r >= R_min:
            mean = float(np.mean(diffs))
            std = float(np.std(diffs, ddof=1)) if r > 1 else 0.0

            if std == 0.0:
                if mean < -tau:
                    return True, r
                if mean >  tau:
                    return False, r
            else:
                hw = float(t_dist.ppf(1.0 - alpha/2.0, r-1) * std / np.sqrt(r))
                if mean + hw < -tau:
                    return True, r
                if mean - hw >  tau:
                    return False, r

    return (np.mean(diffs) < -tau), len(diffs)

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode('utf-8')).hexdigest()

def _ind_key(ind) -> str:
    try:
        return _sha1(str(ind[0]) + '|' + str(ind[1]))
    except Exception:
        return _sha1(repr(ind[0]) + '|' + repr(ind[1]))

def _eval_one_rep_cached(toolbox, ind, rep_seed, cache: dict):
    key = (_ind_key(ind), int(rep_seed))
    if key in cache:
        return cache[key]
    f0, f1 = toolbox.evaluate_one_rep(ind, seed=int(rep_seed))
    cache[key] = (float(f0), float(f1))
    return cache[key]

def update_ideal(z, f):
    z[0] = min(z[0], f[0])
    z[1] = min(z[1], f[1])

def normalize(f, z, scale=None, eps=1e-12):
    if scale is None:
        return np.maximum(0.0, f - z) 
    else:
        s = np.maximum(scale, eps)
        return np.maximum(0.0, (f - z) / s)

def normalize_weights_inplace(lambdas, eps=1e-12):
    for i in range(len(lambdas)):
        lam = np.asarray(lambdas[i], dtype=float)
        lam = np.maximum(lam, eps)
        lambdas[i] = lam / lam.sum()

def compute_scale_from_population(population, eps=1e-12):
    F = np.array([ind.fitness.values for ind in population], dtype=float)
    z = F.min(axis=0)
    nadir = F.max(axis=0)
    scale = np.maximum(nadir - z, eps)
    return z, scale

def g_tchebycheff_abs(f, lam, z, scale=None, rho=1e-6, eps=1e-12):
    f = np.asarray(f, dtype=float)
    z = np.asarray(z, dtype=float)
    lam = np.asarray(lam, dtype=float)
    lam = np.maximum(lam, eps); lam = lam / lam.sum()
    if scale is None:
        fn = np.abs(f - z)
    else:
        s = np.maximum(np.asarray(scale, dtype=float), eps)
        fn = np.abs((f - z) / s)
    g_max = np.max(lam * fn)
    return float(g_max + rho * np.sum(lam * fn))

def select_parents_idx(k, neighbors, K, delta=0.85, n_parents=2):
    pool = neighbors[k] if random.random() < delta else list(range(K))
    if k not in pool:
        pool = pool + [k]
    return random.sample(pool, n_parents)

def decide_replace(child, old, lambdaj, z, scale, rep_seeds, toolbox, cache,
                   R_min=2, R_max=8, alpha=0.05, tau=0.0,
                   penalty_weight=None, penalty_func=None):
    diffs = []
    for r, s in enumerate(rep_seeds[:R_max], start=1):
        fc0, fc1 = _eval_one_rep_cached(toolbox, child, s, cache)
        fo0, fo1 = _eval_one_rep_cached(toolbox, old,   s, cache)
        gc = g_tchebycheff_abs((fc0, fc1), lambdaj, z, scale)
        go = g_tchebycheff_abs((fo0, fo1), lambdaj, z, scale)

        if penalty_weight is not None and penalty_func is not None:
            gc += float(penalty_weight) * float(penalty_func(child))
            go += float(penalty_weight) * float(penalty_func(old))

        d = gc - go
        diffs.append(d)

        if r >= R_min:
            mean = float(np.mean(diffs))
            if r == 1:
                std = 0.0
            else:
                std  = float(np.std(diffs, ddof=1))
            if std == 0.0:
                if mean < -tau:
                    return True, r
                if mean >  tau:
                    return False, r
            else:
                hw = float(t_dist.ppf(1.0 - alpha/2.0, r-1) * std / np.sqrt(r))
                if mean + hw < -tau:
                    return True, r
                if mean - hw >  tau:
                    return False, r
    return (np.mean(diffs) < -tau), len(diffs)

def varAnd(population, toolbox, cxpb, mutpb, reppb):
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb=cxpb/(cxpb+mutpb+reppb)
    new_mutpb=mutpb/(cxpb+mutpb+reppb)+new_cxpb
    i = 1
    while i < len(offspring):
        randomValue = random.random()
        if randomValue < new_cxpb:
            if (offspring[i - 1] == offspring[i]) :
                offspring[i - 1], = toolbox.mutate(offspring[i - 1])
                offspring[i], = toolbox.mutate(offspring[i])
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        elif new_cxpb <= randomValue < new_mutpb:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
        else:
            del offspring[i].fitness.values
            i = i + 1
    return offspring

def sortPopulation(toolbox, population):
    populationCopy = [toolbox.clone(ind) for ind in population]
    popsize = len(population)

    for j in range(popsize):
        sign = False
        for i in range(popsize-1-j):
            sum_fit_i = np.sum(populationCopy[i].fitness.values)
            sum_fit_i_1 = np.sum(populationCopy[i+1].fitness.values)
            if sum_fit_i > sum_fit_i_1:
                populationCopy[i], populationCopy[i+1] = populationCopy[i+1], populationCopy[i]
                sign = True
        if not sign:
            break
    return populationCopy

def crowding_distance(toolbox, population,fronts):
    distances = {}
    num_objectives = 2
    for front in fronts:
        if not front or len(front) == 0:
            continue
        for idx in front:
            distances[idx] = 0.0
        for m in range(num_objectives):
            front_sorted = sorted(front, key=lambda idx: population[idx].fitness.values[m])
            min_value = population[front_sorted[0]].fitness.values[m]
            max_value = population[front_sorted[-1]].fitness.values[m]
            distances[front_sorted[0]] = float('inf')
            distances[front_sorted[-1]] = float('inf')
            if max_value - min_value == 0:
                continue
            for i in range(1, len(front_sorted) - 1):
                prev = population[front_sorted[i - 1]].fitness.values[m]
                next_ = population[front_sorted[i + 1]].fitness.values[m]
                distances[front_sorted[i]] += (next_ - prev) / (max_value - min_value)
    return distances

def nondominated_sort(toolbox, population):
    populationCopy = [toolbox.clone(ind) for ind in population]
    popsize = len(population)
    fronts = []
    dominated_counts = [0] * popsize
    dominates = [[] for _ in range(popsize)]
    za2 = norm.ppf(0.975)
    for i in range(popsize):
        for j in range(popsize):
            if i==j:
                continue
            win,lose,draw=0,0,0
            fit_i=populationCopy[i].fitness_matrix
            fit_j=populationCopy[j].fitness_matrix
            n_rep = len(fit_i)
            for k in range(n_rep):
                if (fit_i[k][0] < fit_j[k][0] and fit_i[k][1] < fit_j[k][1]):
                    win+=1
                elif (fit_i[k][0] > fit_j[k][0] and fit_i[k][1] > fit_j[k][1]):
                    lose+=1
                else:
                    draw+=1
            pw=win/n_rep
            pl=lose/n_rep
            pd=draw/n_rep
            diff = pw - pl
            var = 2*pw * (1 - pw) / n_rep + 2*pl * (1 - pl) / n_rep-(pw+pl)*(1-pw-pl)/n_rep
            std = math.sqrt(var)
            low=diff-za2*std
            up=diff+za2*std
            if low > 0 and up > 0:
                dominates[i].append(j)
                dominated_counts[j] += 1
    current_front = [idx for idx, count in enumerate(dominated_counts) if count == 0]
    assigned = set(current_front)
    fronts.append(current_front)
    while True:
        next_front = []
        for idx in current_front:
            for dominated in dominates[idx]:
                dominated_counts[dominated] -= 1
                if dominated_counts[dominated] == 0 and dominated not in assigned:
                    next_front.append(dominated)
                    assigned.add(dominated)
        if not next_front:
            break
        fronts.append(next_front)
        current_front = next_front
    return fronts

def penalize_duplicates(population):
    seen = set()
    for ind in population:
        key = _ind_key(ind)
        if key in seen:
            inf_pair = (float('inf'), float('inf'))
            ind.fitness.values = inf_pair
            n_rep = len(getattr(ind, "fitness_matrix", [])) or 1
            ind.fitness_matrix = [inf_pair] * n_rep
        else:
            seen.add(key)

def penalize_duplicates_pc(population,pc_vector,tol=1e-12):
    cleared = 0
    for i in range(len(population)):
        ind_i=population[i]
        vec_i=pc_vector[i]
        if getattr(ind_i.fitness, "values", None):
            vals_i=ind_i.fitness.values
            if any(v==float("inf") for v in vals_i):
                continue
        for j in range(i+1, len(population)):
            ind_j=population[j]
            vec_j=pc_vector[j]
            d=dist(vec_i, vec_j)
            if d<= tol:
                already_inf = any(v == float("inf") for v in getattr(ind_j.fitness, "values", ()))
                if not already_inf:
                    inf_pair=(float('inf'), float('inf'))
                    ind_j.fitness.values=inf_pair
                    n_rep=len(getattr(ind_i, "fitness_matrix",[])) or 1
                    ind_j.fitness_matrix=[inf_pair] * n_rep
                    cleared += 1

def sample_points(lst, n=20, replace=False, seed=None):
    rng = random.Random(seed) if seed is not None else random
    if not lst:
        return []
    if replace:
        return [rng.choice(lst) for _ in range(n)]
    k = min(n, len(lst))
    return rng.sample(lst, k)

def _safe_avg(values):
    return sum(values) / max(1, len(values))

def _rou_feats(params):
    now = params["now"] 
    arrival_time = params["arrival_time"]
    due_time = params["due_time"]
    op_index = params["op_index"]
    workloads = params["workloads"]
    cm = params["candidate_machine"]
    rate = cm["rate"]
    proj_avail = cm["proj_available_time"]
    queue_len = cm["queue_len"]
    queue_pt_list = cm["queue_pt_list"]

    NIQ = queue_len
    MWT = max(0.0, proj_avail - now)
    PT  = workloads[op_index] / rate

    if params["next_op"]["exists"] and params["next_op"]["workload"] is not None:
        w_next = params["next_op"]["workload"]
        rates_next = [r["rate"] for r in params["next_op"]["rates"]]
        NPT = _safe_avg([w_next / r for r in rates_next]) if rates_next else 0.0
    else:
        NPT = 0.0

    WIQ = sum(queue_pt_list)
    OWT = 0.0
    TIS = now - arrival_time

    WKR = 0.0
    for item in params["wk_ops_for_WKR"]:
        w = item["workload"]; rates = [r["rate"] for r in item["rates"]]
        WKR += _safe_avg([w / r for r in rates]) if rates else 0.0

    NOR = params["remaining_ops_count"]
    SLACK = due_time - now

    return {"NIQ": NIQ, "MWT": MWT, "PT": PT, "NPT": NPT, "WIQ": WIQ,"OWT": OWT, "TIS": TIS, "WKR": WKR, "NOR": NOR, "SLACK": SLACK}

def _seq_feats(params):
    now = params["now"]
    arrival_time = params["arrival_time"]
    due_time = params["due_time"]
    ready_time = params["ready_time"]
    op_index = params["op_index"]
    workloads = params["workloads"]
    m = params["machine"]
    rate = m["rate"]
    proj_avail = m["proj_available_time"]
    queue_pt_list = m["queue_pt_list"]

    NIQ = m["queue_len"]
    MWT = max(0.0, proj_avail - now)
    PT  = workloads[op_index] / rate

    if params["next_op"]["exists"] and params["next_op"]["workload"] is not None:
        w_next = params["next_op"]["workload"]
        rates_next = [r["rate"] for r in params["next_op"]["rates"]]
        NPT = _safe_avg([w_next / r for r in rates_next]) if rates_next else 0.0
    else:
        NPT = 0.0

    WIQ = sum(queue_pt_list)
    OWT = now - ready_time
    TIS = now - arrival_time

    WKR = 0.0
    for item in params["future_ops"]:
        w = item["workload"]; rates = [r["rate"] for r in item["rates"]]
        WKR += _safe_avg([w / r for r in rates]) if rates else 0.0

    NOR = params["remaining_ops_count"]
    SLACK = due_time - now

    return {"NIQ": NIQ, "MWT": MWT, "PT": PT, "NPT": NPT, "WIQ": WIQ,"OWT": OWT, "TIS": TIS, "WKR": WKR, "NOR": NOR, "SLACK": SLACK}

def build_grouped_statistics(rou, seq):
    rou_statistic = []
    for rec in rou:
        cand_list = rec.get("candidates", [])
        candidates_features = []
        for cand in cand_list:
            feats = _rou_feats(cand["params"])
            candidates_features.append({
                "machine": cand["machine"],
                **feats
            })
        rou_statistic.append({
            "dec_type": "rou",
            "time": rec["time"],
            "job_id": rec["job_id"],
            "op_index": rec["op_index"],
            "chosen_machine": rec.get("chosen_machine"),
            "candidate_count": len(candidates_features),
            "candidates_features": candidates_features
        })

    seq_statistic = []
    for rec in seq:
        cand_list = rec.get("candidates", [])
        candidates_features = []
        for cand in cand_list:
            feats = _seq_feats(cand["params"])
            candidates_features.append({
                "job_id": cand["job_id"],
                "op_index": cand["op_index"],
                **feats
            })
        seq_statistic.append({
            "dec_type": "seq",
            "time": rec["time"],
            "machine": rec["machine"],
            "chosen": rec.get("chosen"),
            "candidate_count": len(candidates_features),
            "candidates_features": candidates_features
        })

    return  rou_statistic, seq_statistic

def semantic_step_window(pc_vector, gen, ngen, k=None, neighbors=None):
    X = np.asarray(pc_vector, float)
    if k is not None and neighbors is not None and neighbors[k]:
        idxs = neighbors[k]
        anchor = X[k][None, :]
        D = np.linalg.norm(X[idxs] - anchor, axis=1)
    else:
        diffs = X[:, None, :] - X[None, :, :]
        D = np.linalg.norm(diffs, axis=2)
        D = D[np.triu_indices_from(D, k=1)]
    q25 = np.quantile(D, 0.25) if D.size else 2.5
    q85 = np.quantile(D, 0.85) if D.size else 8.0

    do_stage_interp=True
    if do_stage_interp:
        t = gen / max(1, float(ngen))
        base_min, base_max = 3.5, 9.0
        data_min, data_max = float(q25), float(q85)
        tau_min = (1 - t) * base_min + t * data_min
        tau_max = (1 - t) * base_max + t * data_max
    else:
        tau_min, tau_max = float(q25), float(q85)

    tau_min = float(np.clip(tau_min, 1.8, 8.5))
    tau_max = float(np.clip(tau_max, tau_min + 0.5, 9.5))
    return tau_min, tau_max

def semantic_window_hybrid(pc_vector, gen, ngen, k, neighbors, outer_power=0.7):
    tauL_min, tauL_max = semantic_step_window(pc_vector, gen, ngen, k=k, neighbors=neighbors)
    tauG_min, tauG_max = semantic_step_window(pc_vector, gen, ngen, k=None, neighbors=None)
    t = gen / max(1., float(ngen))
    w = min(1., max(0., t)) ** outer_power
    tau_min = (1 - w) * tauG_min + w * tauL_min
    tau_max = (1 - w) * tauG_max + w * tauL_max
    eps = 0.5
    tau_min = max(1.8, tau_min); tau_max = max(tau_min + eps, min(9.5, tau_max))
    return tau_min, tau_max

def _pc_standardize(X, eps=1e-12):
    X = np.asarray(X, float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd = np.maximum(sd, eps)
    return (X - mu) / sd

def _pc_knn_neighbors(pc_vector, k):
    X = np.asarray(pc_vector, float)
    n = len(X)
    neighbors = [[] for _ in range(n)]
    if n <= 1:
        return neighbors
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    k = max(1, min(k, n-1))
    idxs = np.argpartition(D, kth=k-1, axis=1)[:, :k]
    for i in range(n):
        row = idxs[i]
        row_sorted = row[np.argsort(D[i, row])]
        neighbors[i] = list(map(int, row_sorted))
    return neighbors

def _adaptive_k(gen, ngen, k_low, k_high):
    t = gen / max(1.0, float(ngen))
    return int(round(k_low + (k_high - k_low) * t))

def _normalize_objectives_matrix(F, z, scale, eps=1e-12):
    F = np.asarray(F, float)
    z = np.asarray(z, float)
    s = np.maximum(np.asarray(scale, float), eps)
    return np.maximum(0.0, (F - z) / s)

def _estimate_local_direction(F_tilde, eps=1e-12):
    Ft = np.asarray(F_tilde, float)

    mask = np.isfinite(Ft).all(axis=1)
    Ft = Ft[mask]
    if Ft.shape[0] < 2:
        return np.array([0.5, 0.5], float)

    mu = Ft.mean(axis=0, keepdims=True)
    Ft = Ft - mu
    if not np.all(np.isfinite(Ft)) or np.all(np.abs(Ft) < eps):
        return np.array([0.5, 0.5], float)

    try:
        U, S, Vt = np.linalg.svd(Ft, full_matrices=False)
        d = Vt[0]
    except np.linalg.LinAlgError:
        C = Ft.T @ Ft
        w, v = np.linalg.eigh(C)
        d = v[:, int(np.argmax(w))]

    if not np.all(np.isfinite(d)):
        return np.array([0.5, 0.5], float)
    n = float(np.linalg.norm(d))
    if n < eps:
        return np.array([0.5, 0.5], float)
    return d / n

def _update_lambdas_from_neighborhood(population, neighbors, z, scale, lambdas, beta=0.2):
    K = len(population)
    for j in range(K):
        idxs = list(neighbors[j]) + [j]
        F_loc = np.array([population[i].fitness.values for i in idxs], float)

        finite_mask = np.isfinite(F_loc).all(axis=1)
        F_loc = F_loc[finite_mask]
        if F_loc.shape[0] < 2:

            continue

        F_tilde = _normalize_objectives_matrix(F_loc, z, scale)
        d = _estimate_local_direction(F_tilde)          
        lam_new = np.abs(d[::-1])                       
        lam_new = np.maximum(lam_new, 1e-12)
        lam_new = lam_new / lam_new.sum()

        lam_old = np.asarray(lambdas[j], float)
        lambdas[j] = ((1.0 - beta) * lam_old + beta * lam_new)

    normalize_weights_inplace(lambdas)

def _pairwise_median_distance(X):
    X = np.asarray(X, float)
    if len(X) < 2:
        return 1.0
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    D = D[np.triu_indices_from(D, k=1)]
    if D.size == 0:
        return 1.0
    return float(np.median(D))

def _compute_pc_density(pc_vector, neighbors, sigma, scope='neighbors'):
    X = np.asarray(pc_vector, float)
    K = len(X)
    Delta = np.zeros(K, dtype=float)
    if K == 0:
        return Delta
    if scope == 'global':

        for i in range(K):
            diff = X - X[i]
            d2 = np.sum(diff * diff, axis=1)
            d2[i] = np.inf
            Delta[i] = float(np.sum(np.exp(-d2 / (sigma * sigma))))
    else:

        for i in range(K):
            s = 0.0
            for j in neighbors[i]:
                d2 = float(np.sum((X[i] - X[j])**2))
                s += math.exp(-d2 / (sigma * sigma))
            Delta[i] = s
    return Delta

def _eta_schedule(gen, ngen, eta0=1e-3, gamma=1.5):
    t = gen / max(1.0, float(ngen))
    return float(eta0 * ((1.0 - t) ** gamma))

def eaSimple(population, toolbox, cxpb, mutpb, reppb, delta, nr, elitism, ngen, seedRotate, rd, K, lambdas, neighbors, if_statistics,if_pc,if_semantic_mutation, if_rebuild_update,if_pc_penalty,g_func='tch',stats=None,halloffame=None, verbose=__debug__, seed = __debug__,
            T_pc_rewire=10, k_low=10, k_high=25, lambda_beta=0.2,
            eta0=1e-3, gamma=1.5, sigma_mode='median', penalty_scope='neighbors', T_pc_penalty=1):

    assert K == len(lambdas) == len(neighbors)
    normalize_weights_inplace(lambdas)
    pop_size=len(population)
    randomSeed_ngen = []
    for i in range((ngen + 1)):
        randomSeed_ngen.append(np.random.randint(2000000000))

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals', 
                    'f0_avg','f0_std','f0_min','f0_max',
                    'f1_avg','f1_std','f1_min','f1_max']
    min_fitness = []
    best_ind_all_gen = []

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    rd['seed'] = randomSeed_ngen[0]

    if if_pc:
        _,_,_,_,rou,seq=jobshop_pc.main(12345,True)
        rou_gt3 = [res for res in rou if len(res.get("candidates", [])) == 3]
        seq_gt3 = [rec for rec in seq if len(rec.get("candidates", [])) == 3]
        rou_decision_point=sample_points(rou_gt3, n=20, replace=False)
        seq_decision_point=sample_points(seq_gt3, n=20, replace=False)
        rou_statistic, seq_statistic= build_grouped_statistics(rou_decision_point, seq_decision_point)
        rou_node_data=[]
        seq_node_data=[]
        for k in range(len(rou_statistic)):
            rou_node_data.append(rou_statistic[k].get("candidates_features", []))
            seq_node_data.append(seq_statistic[k].get("candidates_features", []))
        del rou, seq, rou_gt3, seq_gt3
        del rou_decision_point, seq_decision_point
        del rou_statistic, seq_statistic

    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, ret in zip(invalid_ind, fitnesses):
        scores, fitness_list = ret
        ind.fitness.values = scores
        ind.fitness_matrix = fitness_list
    z, scale = compute_scale_from_population(population)
    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population),
                f0_avg=record['f0']['avg'], f0_std=record['f0']['std'],
                f0_min=record['f0']['min'], f0_max=record['f0']['max'],
                f1_avg=record['f1']['avg'], f1_std=record['f1']['std'],
                f1_min=record['f1']['min'], f1_max=record['f1']['max'])
    if verbose:
        print(logbook.stream)

    pc_vector = None
    pc_density = None
    pc_density_eta = 0.0
    idx_parent_of_child = []

    for gen in range(1, ngen + 1):
        if seedRotate:
            rd['seed'] = randomSeed_ngen[gen]

        fitnesses = toolbox.map(toolbox.evaluate, population)
        for ind, ret in zip(population, fitnesses):
            scores, fitness_list = ret
            ind.fitness.values = scores
            ind.fitness_matrix = fitness_list
        z, scale = compute_scale_from_population(population)

        if if_pc:
            pop_fit_gen = [ind.fitness.values[0]+ind.fitness.values[1] for ind in population]
            best_index_gen = np.argmin(pop_fit_gen)
            pc_vector = computer_PC(population,best_index_gen,rou_node_data,seq_node_data)
            penalize_duplicates_pc(population,pc_vector)
        else:
            penalize_duplicates(population)

        if if_pc and if_rebuild_update and (gen % T_pc_rewire == 0):
            X_std = _pc_standardize(pc_vector)
            k_now = _adaptive_k(gen, ngen, k_low, k_high)
            new_neighbors = _pc_knn_neighbors(X_std, k=k_now)
            neighbors[:] = new_neighbors
            _update_lambdas_from_neighborhood(population, neighbors, z, scale, lambdas, beta=lambda_beta)
            normalize_weights_inplace(lambdas)

        if if_pc and if_pc_penalty and (gen % T_pc_penalty == 0):

            if sigma_mode == 'median':
                sigma = max(1e-6, _pairwise_median_distance(pc_vector))
            else:

                sigma = max(1e-6, _pairwise_median_distance(pc_vector))

            scope = 'global' if penalty_scope == 'global' else 'neighbors'
            pc_density = _compute_pc_density(pc_vector, neighbors, sigma, scope=scope)
            pc_density_eta = _eta_schedule(gen, ngen, eta0=eta0, gamma=gamma)

        children = []
        k_index = []
        idx_parent_of_child = []
        for k in range(K):
            tau_min, tau_max = semantic_window_hybrid(pc_vector, gen, ngen, k, neighbors, outer_power=0.7) if if_pc else (3.5,9.0)
            i1, i2 = select_parents_idx(k, neighbors, K, delta=delta, n_parents=2)
            p1, p2 = population[i1], population[i2]
            c = toolbox.clone(p1)
            if random.random() < cxpb:
                c, _ = toolbox.mate(toolbox.clone(p1), toolbox.clone(p2))
            if random.random() < mutpb:
                if if_pc and if_semantic_mutation:
                    pc_p1=pc_vector[i1]
                    center=0.5*(tau_min+tau_max)
                    accepted=None
                    best=None
                    for l in range(5):
                        temp_combine_population=[]
                        cand, = toolbox.mutate(toolbox.clone(c))
                        temp_combine_population[:]=[toolbox.clone(population[i1]),toolbox.clone(cand),toolbox.clone(population[best_index_gen])]
                        pc_combine=computer_PC(temp_combine_population,2,rou_node_data,seq_node_data)
                        pc_c=pc_combine[1]
                        d=float(dist(pc_p1,pc_c))
                        if tau_min<=d<=tau_max:
                            accepted=cand
                            break
                        if (best is None) or (abs(d-center)<abs(best[1]-center)):
                            best=(c,d)
                    if accepted is not None:
                        c=accepted
                    elif best is not None:
                        c=best[0]
                    else:
                        c, = toolbox.mutate(toolbox.clone(c))
                else:
                    c, = toolbox.mutate(toolbox.clone(c))
            if hasattr(c.fitness, "values"):
                del c.fitness.values
            children.append(c)
            k_index.append(k)
            idx_parent_of_child.append(i1)

        if if_statistics:
            R_MIN  = 4        
            R_MAX  = 10       
            ALPHA  = 0.05     
            TAU    = 0.0      
            rep_cache = {}
            base_seed = int(rd['seed'])
            rep_seeds = [base_seed + 10000 + t for t in range(1000)]
            existing_keys = { _ind_key(ind) for ind in population }
            seen_child_keys = set()
            filtered_pairs = []
            for k, c in zip(k_index, children):
                ck = _ind_key(c)
                if ck in existing_keys:
                    continue           
                if ck in seen_child_keys:
                    continue            
                seen_child_keys.add(ck)
                filtered_pairs.append((k, c))
            if filtered_pairs:
                k_index, children = map(list, zip(*filtered_pairs))
            else:
                k_index, children = [], []

            if if_pc and if_pc_penalty and (pc_density is not None):
                key_to_delta = {}

                for idx, ind in enumerate(population):
                    key_to_delta[_ind_key(ind)] = float(pc_density[idx])

                for c, i1 in zip(children, idx_parent_of_child[:len(children)]):
                    key_to_delta[_ind_key(c)] = float(pc_density[i1])
                penalty_func = lambda ind: key_to_delta.get(_ind_key(ind), 0.0)
                penalty_w = pc_density_eta
            else:
                penalty_func = None
                penalty_w = None

            for k, c in zip(k_index, children):
                replaced = 0
                for j in np.random.permutation(neighbors[k]):
                    if population[j].fitness.values==(float('inf'), float('inf')):
                        ok=True
                    else:
                        ok, used = decide_replace_parallel(
                            child=c, old=population[j], lambdaj=lambdas[j],
                            z=z, scale=scale, rep_seeds=rep_seeds,
                            toolbox=toolbox, cache=rep_cache,
                            R_min=R_MIN, R_max=R_MAX, alpha=ALPHA, tau=TAU,
                            penalty_weight=penalty_w, penalty_func=penalty_func
                        )
                    if ok:
                        population[j] = toolbox.clone(c)
                        replaced += 1
                        if replaced >= nr:
                            break
            invalid_after = [ind for ind in population if not ind.fitness.valid or len(getattr(ind.fitness, "values", ())) == 0]
            if invalid_after:
                results = toolbox.map(toolbox.evaluate, invalid_after)
                for ind, ret in zip(invalid_after, results):
                    scores, fit_list = ret
                    ind.fitness.values = scores
                    ind.fitness_matrix = fit_list

        else:
            results = toolbox.map(toolbox.evaluate, children)
            for c, ret in zip(children, results):
                scores, fit_list = ret
                c.fitness.values = scores
                c.fitness_matrix = fit_list
            penalize_duplicates(population + children)

            if if_pc and if_pc_penalty and (pc_density is not None):
                key_to_delta = {}
                for idx, ind in enumerate(population):
                    key_to_delta[_ind_key(ind)] = float(pc_density[idx])
                for c, i1 in zip(children, idx_parent_of_child[:len(children)]):
                    key_to_delta[_ind_key(c)] = float(pc_density[i1])
                penalty_func = lambda ind: key_to_delta.get(_ind_key(ind), 0.0)
                penalty_w = pc_density_eta
            else:
                penalty_func = None
                penalty_w = None

            for k, c in zip(k_index, children):
                replaced = 0
                for j in np.random.permutation(neighbors[k]):
                    gj_child = g_tchebycheff_abs(c.fitness.values, lambdas[j], z, scale)
                    gj_old   = g_tchebycheff_abs(population[j].fitness.values, lambdas[j], z, scale)

                    if penalty_func is not None and penalty_w is not None:
                        gj_child += float(penalty_w) * float(penalty_func(c))
                        gj_old   += float(penalty_w) * float(penalty_func(population[j]))

                    if gj_child < gj_old:
                        population[j] = toolbox.clone(c)
                        replaced += 1
                        if replaced >= nr:
                            break

        z, scale = compute_scale_from_population(population)
        if halloffame is not None:
            halloffame.clear()
            halloffame.update(population)

        pop_fit = [ind.fitness.values[0]+ind.fitness.values[1] for ind in population]
        best_index = np.argmin(pop_fit)
        best_ind_all_gen.append(population[best_index])
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population),
                    f0_avg=record['f0']['avg'], f0_std=record['f0']['std'],
                    f0_min=record['f0']['min'], f0_max=record['f0']['max'],
                    f1_avg=record['f1']['avg'], f1_std=record['f1']['std'],
                    f1_min=record['f1']['min'], f1_max=record['f1']['max'])
        if verbose:
            print(logbook.stream)

        min_fitness.append(min(pop_fit))

        if gen == ngen:
            top_inds_final_gen = population
            top_inds_fitness_final_gen = [population[i].fitness.values[0] + population[i].fitness.values[1] for i in range(len(population))]

    return population, logbook, min_fitness, best_ind_all_gen, top_inds_fitness_final_gen, top_inds_final_gen
