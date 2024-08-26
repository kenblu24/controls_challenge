import os
import time
import inspect
import cma
import yaml
import numpy as np
from .OptimVar import CMAESVarSet
from dataclasses import make_dataclass
from functools import wraps
import pandas as pd
from tqdm import tqdm
import pathlib as pl

CMAES_NAME = "CMAES"
CONFIG_NAME = "optim-config.yaml"
GENOMES_NAME = "genomes.csv"
CHECKPOINT_NAME = "cmaes.pickle"
PRETTY_NAME = "cmaes-pretty.txt"


def _ignore_no_experiment(func):
    @wraps(func)
    def check(self, *args, **kwargs):
        if self.experiment is not None:
            func(self, *args, **kwargs)
    return check


class CMAES:
    def __init__(
        self,
        f=None,
        dvars=None,
        pop_size=10,
        max_iters=None,
        target=0.0,
        num_processes=1,
        stop_detection_method=None,
        show_each_step=False,
        experiment=None,
        round_to_every=None,
        eval_is_deterministic=False,
        tqdm=True,
    ):
        genome_size = 0 if dvars is None else len(dvars)
        self.f = f
        self.x0 = [0.5 for _ in range(genome_size)]
        self.s0 = 0.20
        self.pop = pop_size
        self.bounds = [[0.0 for _ in range(genome_size)], [1.0 for _ in range(genome_size)]]
        self.target = target
        self.n_processes = num_processes
        self.w_stop_method = stop_detection_method
        self.solution_set = {}
        self.show_steps = show_each_step
        self.dvars = dvars
        self.experiment = experiment
        self.max_iters = max_iters
        self.generation = 0
        self.round_to_every = round_to_every
        self.seed = 1
        self.cache_answers = eval_is_deterministic
        self.use_tqdm = tqdm

        # Data collection
        if self.experiment is not None:
            dvar_repr = [(f"p_{i}" if self.dvars is None else self.dvars.names[i], float) for i in range(len(self.x0))]
            # b_names = self.behavior_names()
            b_names = ["error"]
            b_repr = [(b_name, float) for b_name in b_names]
            self.hist_point = make_dataclass(
                "CMAESPoint", [("time", int), ("gen", int), ("pop", int), ("fitness", float)] + dvar_repr + b_repr
            )
            self.history = []
            self.exp_path = self.experiment.add_sub("CMAES")
            with open(self.exp_path / CONFIG_NAME, "w") as out_f:
                yaml.dump(self.parameters_dict(), out_f, default_flow_style=False)

    @property
    def opts(self):
        return {
            "seed": self.seed,
            "popsize": self.pop,
            "bounds": self.bounds,
            "ftarget": self.target,
            "tolfun": 0,
            "tolflatfitness": 500,
            "tolfunhist": 0,
        }

    def minimize(self):
        self.es = es = cma.CMAEvolutionStrategy(self.x0, self.s0, self.opts)
        while not es.stop():
            try:
                parameters = es.ask()
                solutions = self.ask_for_genomes(parameters)
                # if self.cache_answers:
                es.tell(solutions, [self.pull_from_solution_set(s) for s in solutions])
                es.disp()

                self.write_genomes()
                self.write_cma_checkpoint()
                self.write_best()

                self.generation += 1
                if self.show_steps:
                    raise NotImplementedError("Show steps not implemented")
                if self.max_iters and self.generation > self.max_iters:
                    break
            except KeyboardInterrupt:
                break

        es.result_pretty()
        return es.result, es

    @staticmethod
    def round_to_nearest(a, increment):
        return [round(round(a_i / increment) * increment, 2) for a_i in a]

    def sweep_parameters(self, divisions: list[int]):
        if len(divisions) != len(self.x0):
            raise Exception(f"Divisions should be of size {len(self.x0)}. Not {len(divisions)}")  # noqa: EM102
        spaces = []
        for _i, d in enumerate(divisions):
            spaces.append(np.linspace(0, 1, d))
        grid = np.meshgrid(*spaces)
        points = np.array(grid).reshape((len(divisions), -1)).T
        parameters = [self.dvars.from_unit_to_scaled(p) for p in points]
        # processor = MultiWorldSimulation(pool_size=self.n_processes, single_step=False, with_gui=False)

        # configs = [self.g_to_w(parameters[i], parameters[i]) for i in range(len(parameters))]
        # ret = processor.execute(configs, world_stop_condition=self.w_stop_method, batched=True)

        for i, genome in enumerate(parameters):
            _key = genome[0].meta["hash"]
            fitness = self.f(genome)
            # behavior = self.average_behaviors(genome)
            behavior = [fitness]
            self.solution_set[_key] = fitness
            if self.experiment is not None:
                self.history.append(self.hist_point(self.generation, i, fitness, *parameters[i], *behavior))

        if self.experiment is not None:
            df = pd.DataFrame(self.history)
            df.to_csv(self.exp_path / GENOMES_NAME, index=False)

    def ask_for_genomes(self, parameters):
        normalized = parameters
        if self.round_to_every is not None:
            parameters = [self.round_to_nearest(p, increment=self.round_to_every) for p in parameters]
        if self.dvars:
            parameters = [self.dvars.from_unit_to_scaled(p) for p in parameters]

        # batched_worlds = isinstance(configs[0], list)

        # Blocking MultiProcess Execution
        # ret = processor.execute(configs, world_stop_condition=self.w_stop_method, batched=batched_worlds)
        if tqdm:
            results = [self.f(genome) for genome in tqdm(parameters)]
        else:
            results = [self.f(genome) for genome in parameters]
        hashes = [hash(tuple(list(normalized_genome))) for normalized_genome in normalized]
        t = int(time.time())
        for i, (key, genome, result) in enumerate(zip(hashes, parameters, results)):
            fitness = result
            # behavior = self.average_behaviors(unit)
            behavior = [fitness]
            self.solution_set[key] = fitness
            if self.experiment is not None:
                self.history.append(self.hist_point(t, self.generation, i, fitness, *genome, *behavior))
        # print(self.solution_set)
        return normalized

    def pull_from_solution_set(self, x):
        fetch_key = hash(tuple(list(x)))
        retrieval = None
        if fetch_key in self.solution_set:
            retrieval = self.solution_set[fetch_key]

        if retrieval is None:
            return self.f(x)
        else:
            return retrieval

    # def average_behaviors(self, world_set):
    #     behaviors = np.zeros(len(world_set[0].behavior), dtype=float)
    #     for world in world_set:
    #         behaviors += np.array([world.behavior[i].out_average()[1] for i in range(len(world.behavior))])
    #     return behaviors / len(world_set)

    # def behavior_names(self):
    #     # Extract a random world
    #     world_set = self.g_to_w(self.x0, self.x0)
    #     if isinstance(world_set, list):
    #         world = world_set[0]
    #     else:
    #         world = world_set
    #     behavior_names = [world.behavior[i].name for i in range(len(world.behavior))]
    #     return behavior_names

    @_ignore_no_experiment
    def write_cma_checkpoint(self):
        with open(self.exp_path / CHECKPOINT_NAME, "wb") as f:
            f.write(self.es.pickle_dumps())

    @_ignore_no_experiment
    def write_genomes(self):
        df = pd.DataFrame(self.history)
        df.to_csv(self.exp_path / GENOMES_NAME, index=False)

    @_ignore_no_experiment
    def write_best(self):
        pass
        # raise NotImplementedError("")
        # with open(self.exp_path / PRETTY_NAME, "w") as f:
        #     f.write(str(self.es.result_pretty()))

    def parameters_dict(self):
        try:
            func_source = inspect.getsource(self.f).split("\n")
        except TypeError as err:
            if 'partial' in str(err):
                func_source = [repr(self.f)]
                func_source += inspect.getsource(self.f.func).split("\n")
            else:
                raise
        return {
            "fitness_func": func_source,
            "initial_genome": self.x0,
            "initial_sigma": self.s0,
            "population_size": self.pop,
            "bounds": self.bounds,
            "target": self.target,
            "n_processes": self.n_processes,
            "w_stop_method": inspect.getsource(self.w_stop_method).split("\n") if self.w_stop_method is not None else None,
            "show_steps": self.show_steps,
            "dvars": self.dvars.as_ordered_dict() if self.dvars else None,
        }
