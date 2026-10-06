from __future__ import annotations

import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional

import simpy

from tree_calculate import GP_evolve_S, GP_pair_S_test

@dataclass
class DFJSPConfig:
    seed: int = 6666

    num_machines: int = 10

    total_jobs: int = 50
    warmup_jobs: int = 4
    ops_per_job: int = 10
    workload_min: int = 100
    workload_max: int = 1000

    rate_min: int = 10
    rate_max: int = 15

    due_factor: float = 1.5

    util_level: float = 0.85

    arrival_rate_override: Optional[float] = None

    interarrival_mean: Optional[float] = None
    candidate_min: int = 2
    candidate_max: Optional[int] = None
    weight_values: tuple = (1, 2, 4)
    weight_probs: tuple = (0.2, 0.6, 0.2)
    poll_interval: float = 0.1
    simulation_stop_mode: str = "all_jobs_completed"

    verbose: bool = False

    rou_tree: Optional[object] = None
    seq_tree: Optional[object] = None
    ifPrint: bool = False
    ifTest: bool = False

@dataclass
class Task:
    job_id: int
    op_index: int
    workload: int
    machine_name: str
    pt: float
    done: simpy.events.Event
    ready_time: float
    due_time: float
    workloads_job: List[int]
    op_machine_map_job: Dict[int, List[str]]
    arrival_time: float
    machines: Dict[str, "Machine"]

class Machine:
    def __init__(self, env: simpy.Environment, name: str, rate: int, cfg: DFJSPConfig):
        self.env = env
        self.name = name
        self.rate = rate
        self.queue: simpy.Store[Task] = simpy.Store(env)
        self.total_proc_time = 0.0
        self.completed_ops = 0
        self.proj_available_time: float = 0.0
        self.total_idle_time = 0.0
        self.last_busy_time = 0.0
        self.energy_consumption = 0.0
        self.cfg = cfg
        self._proc = env.process(self._worker())

    def _worker(self):
        while True:
            if self.queue.items:
                current_time = self.env.now

                prioritized_tasks = []
                for task in list(self.queue.items):
                    data = Machine.calc_queue_status(
                        machines=task.machines,
                        op_index=task.op_index,
                        workloads=task.workloads_job,
                        op_machine_map=task.op_machine_map_job,
                        machine_name=task.machine_name,
                        current_time=current_time,
                        arrival_time=task.arrival_time,
                        due_time=task.due_time,
                        ready_time=task.ready_time,
                    )
                    priority = GP_pair_S_test(data, self.cfg.seq_tree) if self.cfg.ifTest else GP_evolve_S(
                        data, self.cfg.seq_tree
                    )
                    prioritized_tasks.append((priority, task))

                prioritized_tasks.sort(key=lambda x: x[0])
                _, selected_task = prioritized_tasks[0]
                self.queue.items.remove(selected_task)

                idle_time = max(0, self.env.now - self.last_busy_time)
                self.total_idle_time += idle_time
                self.energy_consumption += idle_time * 1

                yield self.env.timeout(selected_task.pt)
                self.total_proc_time += selected_task.pt
                self.completed_ops += 1
                self.energy_consumption += selected_task.pt * 4

                self.last_busy_time = self.env.now

                if not self.queue.items:
                    self.proj_available_time = self.env.now

                if not selected_task.done.triggered:
                    selected_task.done.succeed(self.env.now)
            else:
                yield self.env.timeout(self.cfg.poll_interval)

    def enqueue(self, task: Task):
        self.queue.put(task)

    @staticmethod
    def calc_queue_status(
        machines: Dict[str, "Machine"],
        op_index: int,
        workloads: List[int],
        op_machine_map: Dict[int, List[str]],
        machine_name: str,
        current_time: float,
        arrival_time: float,
        due_time: float,
        ready_time: float,
    ) -> List[float]:
        machine = machines[machine_name]

        NIQ = len(machine.queue.items)
        MWT = max(0, machine.proj_available_time - current_time)
        rate = machine.rate
        PT = workloads[op_index] / rate

        NPT = 0.0
        if op_index + 1 < len(op_machine_map):
            next_machine_names = op_machine_map[op_index + 1]
            for m_name in next_machine_names:
                m = machines[m_name]
                NPT += workloads[op_index + 1] / m.rate
            NPT /= max(1, len(next_machine_names))

        WIQ = sum(op.pt for op in machine.queue.items)
        OWT = current_time - ready_time
        TIS = current_time - arrival_time

        WKR = 0.0
        for i in range(op_index + 1, len(workloads)):
            wl = workloads[i]
            candidates = op_machine_map[i]
            avg_pt = sum(wl / machines[n].rate for n in candidates) / len(candidates)
            WKR += avg_pt

        NOR = len(workloads) - op_index - 1
        SLACK = due_time - current_time

        return [NIQ, MWT, PT, NPT, WIQ, OWT, TIS, WKR, NOR, SLACK]

    @staticmethod
    def calc_machine_status(
        machines: Dict[str, "Machine"],
        op_index: int,
        workloads: List[int],
        op_machine_map: Dict[int, List[str]],
        machine: "Machine",
        next_machine_names: List[str],
        current_time: float,
        arrival_time: float,
        due_time: float,
    ) -> List[float]:
        NIQ = len(machine.queue.items)
        MWT = max(0, machine.proj_available_time - current_time)
        PT = workloads[op_index] / machine.rate

        NPT = 0.0
        if next_machine_names:
            for m_name in next_machine_names:
                m = machines[m_name]
                NPT += workloads[op_index + 1] / m.rate
            NPT /= len(next_machine_names)

        WIQ = sum(op.pt for op in machine.queue.items)
        OWT = 0.0
        TIS = current_time - arrival_time

        WKR = 0.0
        for i in range(op_index, len(workloads)):
            wl = workloads[i]
            candidates = op_machine_map[i]
            avg_pt = sum(wl / machines[n].rate for n in candidates) / len(candidates)
            WKR += avg_pt

        NOR = len(workloads) - op_index - 1
        SLACK = due_time - current_time

        return [NIQ, MWT, PT, NPT, WIQ, OWT, TIS, WKR, NOR, SLACK]

class DDFJSP:
    def __init__(self, env: simpy.Environment, cfg: DFJSPConfig):
        self.env = env
        self.cfg = cfg
        self._validate_config()
        self.rng = random.Random(cfg.seed)

        self.machines: Dict[str, Machine] = {}
        for i in range(cfg.num_machines):
            name = f"M{i+1}"
            rate = self.rng.randint(cfg.rate_min, cfg.rate_max)
            self.machines[name] = Machine(env, name, rate, cfg)

        self.avg_inv_rate = self._global_avg_inv_rate()

        self.u_mean = self._compute_u_mean()
        self.Pm = min(1.0, self.cfg.ops_per_job / max(1, self.cfg.num_machines))
        self.p_util = self.cfg.util_level
        self.arrival_rate = cfg.arrival_rate_override or self._compute_poisson_lambda()

        self.jobs_generated = 0
        self.jobs_completed_total = 0
        self.measured_jobs = 0

        self.tardiness_measured: List[float] = []
        self.weighted_tardiness_measured: List[float] = []

        self.flow_times_measured: List[float] = []
        self.weighted_flow_times_measured: List[float] = []

        self.max_flow_time: float = 0.0
        self.max_tardiness: float = 0.0
        self.max_weighted_flow_time: float = 0.0
        self.max_weighted_tardiness: float = 0.0

        self.all_done = env.event()
        self.env.process(self.job_arrival_process())

    def _validate_config(self):
        if self.cfg.num_machines < 2:
            raise ValueError("num_machines must be at least 2")
        candidate_max = self.cfg.candidate_max or self.cfg.num_machines
        if not 1 <= self.cfg.candidate_min <= candidate_max <= self.cfg.num_machines:
            raise ValueError("candidate_min/max must satisfy 1 <= min <= max <= num_machines")
        if len(self.cfg.weight_values) != len(self.cfg.weight_probs) or not self.cfg.weight_values:
            raise ValueError("weight_values and weight_probs must have the same non-zero length")
        if abs(sum(self.cfg.weight_probs) - 1.0) > 1e-9:
            raise ValueError("weight_probs must sum to 1")
        if self.cfg.interarrival_mean is not None and self.cfg.interarrival_mean <= 0:
            raise ValueError("interarrival_mean must be positive")
        if self.cfg.poll_interval <= 0:
            raise ValueError("poll_interval must be positive")

    def _global_avg_inv_rate(self) -> float:
        inv_rates = [1.0 / m.rate for m in self.machines.values()]
        return sum(inv_rates) / len(inv_rates)

    def _compute_u_mean(self) -> float:
        w_min, w_max = self.cfg.workload_min, self.cfg.workload_max
        ew = (w_min + w_max) / 2.0
        return ew * self.avg_inv_rate

    def _compute_poisson_lambda(self) -> float:
        lam = (self.u_mean * self.Pm) / max(self.p_util, 1e-9)
        return max(lam, 1e-9)

    def _draw_job_weight(self) -> int:
        r = self.rng.random()
        values = self.cfg.weight_values
        probs = self.cfg.weight_probs
        cumulative = 0.0
        for value, probability in zip(values, probs):
            cumulative += probability
            if r < cumulative:
                return value
        return values[-1]

    def _gen_workloads(self) -> List[int]:
        return [self.rng.randint(self.cfg.workload_min, self.cfg.workload_max) for _ in range(self.cfg.ops_per_job)]

    def _due_date(self, arrival_time: float, workloads: List[int]) -> float:
        exp_total_pt = sum(workloads) * self.avg_inv_rate
        return arrival_time + self.cfg.due_factor * exp_total_pt

    def job_arrival_process(self):

        beta = self.cfg.interarrival_mean
        if beta is None:
            beta = 50 / (self.cfg.util_level * 10)

        while self.jobs_generated < self.cfg.total_jobs:
            gap = self.rng.expovariate(1.0 / beta)
            yield self.env.timeout(gap)

            self.jobs_generated += 1
            job_id = self.jobs_generated
            arrival = self.env.now
            weight = self._draw_job_weight()
            workloads = self._gen_workloads()
            due = self._due_date(arrival, workloads)

            op_machine_map: Dict[int, List[str]] = {}
            machine_names = list(self.machines.keys())
            for op_index in range(self.cfg.ops_per_job):
                num_candidates = self.rng.randint(
                    self.cfg.candidate_min,
                    self.cfg.candidate_max or len(machine_names),
                )
                op_machine_map[op_index] = self.rng.sample(machine_names, num_candidates)

            self.env.process(
                self.job_process(
                    job_id=job_id,
                    arrival=arrival,
                    due=due,
                    weight=weight,
                    workloads=workloads,
                    op_machine_map=op_machine_map,
                )
            )

    def job_process(
        self,
        job_id: int,
        arrival: float,
        due: float,
        weight: int,
        workloads: List[int],
        op_machine_map: Dict[int, List[str]],
    ):
        prev_op_done = self.env.event()
        prev_op_done.succeed()

        for op_index, w in enumerate(workloads):
            now = self.env.now

            candidates = op_machine_map[op_index]

            cand_scores = []
            for m_name in candidates:
                m = self.machines[m_name]
                next_names = op_machine_map[op_index + 1] if op_index + 1 < len(workloads) else []
                data = Machine.calc_machine_status(
                    machines=self.machines,
                    op_index=op_index,
                    workloads=workloads,
                    op_machine_map=op_machine_map,
                    machine=m,
                    next_machine_names=next_names,
                    current_time=now,
                    arrival_time=arrival,
                    due_time=due,
                )
                score = GP_pair_S_test(data, self.cfg.rou_tree) if self.cfg.ifTest else GP_evolve_S(data, self.cfg.rou_tree)
                cand_scores.append((score, m_name))

            cand_scores.sort(key=lambda x: x[0])
            sel_m_name = cand_scores[0][1]
            sel_machine = self.machines[sel_m_name]

            yield prev_op_done

            est_start = max(self.env.now, sel_machine.proj_available_time)
            pt = w / sel_machine.rate
            est_complete = est_start + pt

            done_event = self.env.event()
            task = Task(
                job_id=job_id,
                op_index=op_index,
                workload=w,
                machine_name=sel_m_name,
                pt=pt,
                done=done_event,
                ready_time=self.env.now,
                due_time=due,
                workloads_job=workloads,
                op_machine_map_job=op_machine_map,
                arrival_time=arrival,
                machines=self.machines,
            )

            sel_machine.proj_available_time = max(sel_machine.proj_available_time, est_complete)
            sel_machine.enqueue(task)
            yield task.done

            prev_op_done = done_event

            if self.cfg.verbose:
                print(f"[{self.env.now:7.2f}] J{job_id} op{op_index+1} on {sel_m_name} rate={sel_machine.rate} w={w} pt={pt:.2f}")

        completion = self.env.now
        flow = completion - arrival
        tard = max(0.0, completion - due)
        wf = weight * flow
        wt = weight * tard

        if job_id > self.cfg.warmup_jobs:

            self.tardiness_measured.append(tard)
            self.weighted_tardiness_measured.append(wt)

            self.flow_times_measured.append(flow)
            self.weighted_flow_times_measured.append(wf)
            self.measured_jobs += 1

        self.jobs_completed_total += 1
        if self.jobs_completed_total == self.cfg.total_jobs and not self.all_done.triggered:
            self.all_done.succeed(True)

        if self.cfg.verbose:
            print(f"[{self.env.now:7.2f}] J{job_id} done flow={flow:.2f} tard={tard:.2f} w={weight}")

    def run(self):
        self.env.run(until=self.all_done)

    def report(self):
        makespan = self.env.now

        if self.measured_jobs > 0:
            self.TTD = sum(self.tardiness_measured)
            self.WTTD = sum(self.weighted_tardiness_measured)
        else:
            self.TTD = 0.0
            self.WTTD = 0.0

        if self.measured_jobs > 0:
            self.max_flow_time = max(self.flow_times_measured) if self.flow_times_measured else 0.0
            self.max_tardiness = max(self.tardiness_measured) if self.tardiness_measured else 0.0
            self.max_weighted_flow_time = max(self.weighted_flow_times_measured) if self.weighted_flow_times_measured else 0.0
            self.max_weighted_tardiness = max(self.weighted_tardiness_measured) if self.weighted_tardiness_measured else 0.0
        else:
            self.max_flow_time = 0.0
            self.max_tardiness = 0.0
            self.max_weighted_flow_time = 0.0
            self.max_weighted_tardiness = 0.0

        self.total_energy = 0.0
        for m in self.machines.values():
            self.total_energy += m.energy_consumption

def main(rou_tree, seq_tree, seed=6666, ifPrint=False, ifTest=False, simulation_config=None):
    simulation_config = simulation_config or {}
    cfg = DFJSPConfig(
        rou_tree=rou_tree,
        seq_tree=seq_tree,
        seed=seed,
        num_machines=simulation_config.get("num_machines", 5),
        total_jobs=simulation_config.get("total_jobs", 124),
        warmup_jobs=simulation_config.get("warmup_jobs", 16),
        ops_per_job=simulation_config.get("ops_per_job", 10),
        workload_min=simulation_config.get("workload_min", 100),
        workload_max=simulation_config.get("workload_max", 1000),
        rate_min=simulation_config.get("rate_min", 10),
        rate_max=simulation_config.get("rate_max", 15),
        due_factor=simulation_config.get("due_factor", 1.5),
        util_level=simulation_config.get("util_level", 0.85),
        arrival_rate_override=simulation_config.get("arrival_rate_override"),
        interarrival_mean=simulation_config.get("interarrival_mean"),
        candidate_min=simulation_config.get("candidate_min", 2),
        candidate_max=simulation_config.get("candidate_max"),
        weight_values=tuple(simulation_config.get("weight_values", (1, 2, 4))),
        weight_probs=tuple(simulation_config.get("weight_probs", (0.2, 0.6, 0.2))),
        poll_interval=simulation_config.get("poll_interval", 0.1),
        simulation_stop_mode=simulation_config.get("simulation_stop_mode", "all_jobs_completed"),
        verbose=False,
        ifPrint=ifPrint,
        ifTest=ifTest,
    )
    env = simpy.Environment()
    sim = DDFJSP(env, cfg)
    sim.run()
    sim.report()

    return sim.max_flow_time,sim.max_tardiness,sim.max_weighted_flow_time,sim.max_weighted_tardiness
