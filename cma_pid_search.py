from CMAES import CMAES, CMAESVarSet
from novel_swarms.results.Experiment import Experiment
import numpy as np
import random

DECISION_VARS = CMAESVarSet({
    "kp": [-0.1, 0.2],
    "ki": [-0.1, 0.2],
    "kd": [-0.2, 0.2],
    "kdd": [-0.2, 0.2],
})

round_vars_to_nearest = None
# round_vars_to_nearest = 1 / (int(args.discrete_bins) - 1)

maxsamples = 72


def fitness(genome):
    import pid_with_genome as prog
    files_to_test = [rnd.randrange(0, 20000 - 1) for _ in range(maxsamples)]
    # genome = (0.12, 0.115, 0.005, -0.0008)
    results = prog.test_genome(genome, files_to_test, max_workers=None, tqdm=False)
    # print(results)
    return np.mean([result["total_cost"] for result in results])


if __name__ == "__main__":

    rnd = random.Random(42)
    # only run this line if __main__ or else
    exp = Experiment(root="out", title=None)

    cmaes = CMAES(
        fitness,
        dvars=DECISION_VARS,
        # num_processes=args.processes,
        show_each_step=False,
        target=0,
        experiment=exp,
        max_iters=100,
        pop_size=10,
        # round_to_every=round_vars_to_nearest,
    )
    cmaes.minimize()
