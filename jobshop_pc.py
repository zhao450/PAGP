from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import simpy

def get_rt_queue_params(
    machines: Dict[str, "Machine"],
    op_index: int,
    workloads: List[int],
    op_machine_map: Dict[int, List[str]],
    machine_name: str,
    current_time: float,
    arrival_time: float,
    due_time: float,
    ready_time: float,
):
    m = machines[machine_name]
    queue_pt_list = [t.pt for t in m.queue.items]

    next_machines = []
    if op_index + 1 < len(workloads):
        for mn in op_machine_map[op_index + 1]:
            mm = machines[mn]
            next_machines.append({"name": mn, "rate": mm.rate})

    future_ops = []
    for i in range(op_index + 1, len(workloads)):
        cand = op_machine_map[i]
        rates = [{"name": mn, "rate": machines[mn].rate} for mn in cand]
        future_ops.append({"op_index": i, "workload": workloads[i], "candidates": cand[:], "rates": rates})

    return {
        "now": current_time,
        "arrival_time": arrival_time,
        "due_time": due_time,
        "ready_time": ready_time,
        "op_index": op_index,
        "workloads": workloads[:],
        "op_machine_map": {k: v[:] for k, v in op_machine_map.items()},
        "machine": {
            "name": machine_name,
            "rate": m.rate,
            "proj_available_time": m.proj_available_time,
            "queue_len": len(m.queue.items),
            "queue_pt_list": queue_pt_list,
        },
        "next_op": {
            "exists": op_index + 1 < len(workloads),
            "workload": workloads[op_index + 1] if op_index + 1 < len(workloads) else None,
            "candidates": [x["name"] for x in next_machines],
            "rates": next_machines,
        },
        "future_ops": future_ops,
        "remaining_ops_count": len(workloads) - op_index - 1,
    }

def get_rt_machine_params(
    machines: Dict[str, "Machine"],
    op_index: int,
    workloads: List[int],
    op_machine_map: Dict[int, List[str]],
    machine: "Machine",
    next_machine_names: List[str],
    current_time: float,
    arrival_time: float,
    due_time: float,
):
    queue_pt_list = [t.pt for t in machine.queue.items]
    next_rates = [{"name": mn, "rate": machines[mn].rate} for mn in next_machine_names]

    wk_ops = []
    for i in range(op_index, len(workloads)):
        cand = op_machine_map[i]
        rates = [{"name": mn, "rate": machines[mn].rate} for mn in cand]
        wk_ops.append({"op_index": i, "workload": workloads[i], "candidates": cand[:], "rates": rates})

    return {
        "now": current_time,
        "arrival_time": arrival_time,
        "due_time": due_time,
        "op_index": op_index,
        "workloads": workloads[:],
        "op_machine_map": {k: v[:] for k, v in op_machine_map.items()},
        "candidate_machine": {
            "name": machine.name,
            "rate": machine.rate,
            "proj_available_time": machine.proj_available_time,
            "queue_len": len(machine.queue.items),
            "queue_pt_list": queue_pt_list,
        },
        "next_op": {
            "exists": op_index + 1 < len(workloads),
            "workload": workloads[op_index + 1] if op_index + 1 < len(workloads) else None,
            "candidates": [x["name"] for x in next_rates],
            "rates": next_rates,
        },
        "wk_ops_for_WKR": wk_ops,
        "remaining_ops_count": len(workloads) - op_index - 1,
    }

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

    verbose: bool = False
    ifPrint: bool = False

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
    def __init__(self, env: simpy.Environment, name: str, rate: int, cfg: DFJSPConfig, parent_sim: "DDFJSP"):
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
        self.sim = parent_sim
        self._proc = env.process(self._worker())

    def _worker(self):
        while True:
            if self.queue.items:
                current_time = self.env.now

                seq_candidates = []
                for task in list(self.queue.items):
                    params = get_rt_queue_params(
                        machines=task.machines,
                        op_index=task.op_index,
                        workloads=task.workloads_job,
                        op_machine_map=task.op_machine_map_job,
                        machine_name=self.name,
                        current_time=current_time,
                        arrival_time=task.arrival_time,
                        due_time=task.due_time,
                        ready_time=task.ready_time,
                    )
                    seq_candidates.append({
                        "job_id": task.job_id,
                        "op_index": task.op_index,
                        "params": params,
                    })

                selected_task = self.queue.items[0]

                self.sim.seq_decision_point.append({
                    "time": current_time,
                    "machine": self.name,
                    "chosen": {"job_id": selected_task.job_id, "op_index": selected_task.op_index},
                    "candidates": seq_candidates,
                    "policy": "FIFO"
                })

                self.queue.items.pop(0)

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
                yield self.env.timeout(0.1)

    def enqueue(self, task: Task):
        self.queue.put(task)

class DDFJSP:
    def __init__(self, env: simpy.Environment, cfg: DFJSPConfig):
        self.env = env
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)

        self.rou_decision_point: List[dict] = []
        self.seq_decision_point: List[dict] = []

        self.machines: Dict[str, Machine] = {}
        for i in range(cfg.num_machines):
            name = f"M{i+1}"
            rate = self.rng.randint(cfg.rate_min, cfg.rate_max)
            self.machines[name] = Machine(env, name, rate, cfg, parent_sim=self)

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
        if r < 0.2:
            return 1
        elif r < 0.8:
            return 2
        else:
            return 4

    def _gen_workloads(self) -> List[int]:
        return [self.rng.randint(self.cfg.workload_min, self.cfg.workload_max) for _ in range(self.cfg.ops_per_job)]

    def _due_date(self, arrival_time: float, workloads: List[int]) -> float:
        exp_total_pt = sum(workloads) * self.avg_inv_rate
        return arrival_time + self.cfg.due_factor * exp_total_pt

    def job_arrival_process(self):

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
                num_candidates = self.rng.randint(2, len(machine_names))
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
            next_names = op_machine_map[op_index + 1] if op_index + 1 < len(workloads) else []

            rou_candidates = []
            ea_list = []
            for m_name in candidates:
                m = self.machines[m_name]

                params = get_rt_machine_params(
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

                available_time = max(now, m.proj_available_time)
                ea_list.append((available_time, m_name))

                rou_candidates.append({
                    "machine": m_name,
                    "params": params,
                    "ea_available_time": available_time
                })

            ea_list.sort(key=lambda x: (x[0], x[1]))
            sel_m_name = ea_list[0][1]
            sel_machine = self.machines[sel_m_name]

            self.rou_decision_point.append({
                "time": now,
                "job_id": job_id,
                "op_index": op_index,
                "chosen_machine": sel_m_name,
                "candidates": rou_candidates,
                "policy": "EA"
            })

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

        if self.measured_jobs > 0:
            self.TTD = sum(self.tardiness_measured)
            self.WTTD = sum(self.weighted_tardiness_measured)
            self.max_flow_time = max(self.flow_times_measured) if self.flow_times_measured else 0.0
            self.max_tardiness = max(self.tardiness_measured) if self.tardiness_measured else 0.0
            self.max_weighted_flow_time = max(self.weighted_flow_times_measured) if self.weighted_flow_times_measured else 0.0
            self.max_weighted_tardiness = max(self.weighted_tardiness_measured) if self.weighted_tardiness_measured else 0.0
        else:
            self.TTD = 0.0
            self.WTTD = 0.0
            self.max_flow_time = 0.0
            self.max_tardiness = 0.0
            self.max_weighted_flow_time = 0.0
            self.max_weighted_tardiness = 0.0

        self.total_energy = 0.0
        for m in self.machines.values():
            self.total_energy += m.energy_consumption

def main(seed=6666, ifPrint=False):
    cfg = DFJSPConfig(
        seed=seed,
        num_machines=10,
        total_jobs=500,
        warmup_jobs=50,
        ops_per_job=10,
        workload_min=100,
        workload_max=1000,
        rate_min=10,
        rate_max=15,
        due_factor=1.5,
        util_level=0.85,
        arrival_rate_override=None,
        verbose=False,
        ifPrint=ifPrint,
    )
    env = simpy.Environment()
    sim = DDFJSP(env, cfg)
    sim.run()
    sim.report()

    return (sim.max_flow_time, sim.max_tardiness,sim.max_weighted_flow_time,sim.max_weighted_tardiness,sim.rou_decision_point,sim.seq_decision_point)

if __name__ == "__main__":
    main(seed=12345, ifPrint=True)
