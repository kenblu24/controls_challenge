from functools import partial
from CMAES import CMAES, CMAESVarSet
from CMAES.Experiment import Experiment
import numpy as np
import random
import argparse
import yaml

DECISION_VARS = CMAESVarSet({
    "kp": [-0.1, 0.2],
    "ki": [-0.1, 0.2],
    "kd": [-0.2, 0.2],
    "kdd": [-0.2, 0.2],
})

round_vars_to_nearest = None
# round_vars_to_nearest = 1 / (int(args.discrete_bins) - 1)


def fitness(genome, maxsamples, seg_range=None, max_workers=None, rng=None):
    if seg_range is None:
        seg_range = [0, 5000 - 1]
    if rng is None:
        rng = random.Random()
    import pid_with_genome as prog
    # https://stackoverflow.com/questions/9755538/how-do-i-create-a-list-of-random-numbers-without-duplicates
    segs_to_test = rng.sample(range(*seg_range), maxsamples)
    # genome = (0.12, 0.115, 0.005, -0.0008)
    results = prog.test_genome(genome, segs_to_test, max_workers=max_workers, tqdm=False)
    # print(results)
    return np.mean([result["total_cost"] for result in results])


def main(args):
    rng = random.Random(42)
    # only run this line if __main__ or else
    exp = Experiment(root="out", title=None)

    with open(exp.path / "args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    fitness_partial = partial(
        fitness,
        maxsamples=args.maxsamples,
        seg_range=args.seg_range,
        max_workers=args.processes,
        rng=rng,
    )

    cmaes = CMAES(
        fitness_partial,
        dvars=DECISION_VARS,
        # num_processes=args.processes,
        show_each_step=False,
        target=0,
        experiment=exp,
        max_iters=args.epochs,
        pop_size=10,
        # round_to_every=round_vars_to_nearest,
    )
    cmaes.minimize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxsamples", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seg_range", nargs=2, type=int, default=[0, 5000])
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args()

    main(args)
