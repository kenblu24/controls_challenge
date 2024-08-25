from pathlib import Path, PurePosixPath
from controllers import pid
from functools import partial
from hashlib import md5
import tinyphysics
from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm

import random

MODEL_PATH = "./models/tinyphysics.onnx"
DATA_PATH = "./data"


def run_rollout(data_path, controller, model_path, debug=False, seed=None):
    model = TinyPhysicsModel(model_path, debug=debug)
    data_path = str(data_path)
    sim = TinyPhysicsSimulator(model, data_path, controller=controller, debug=debug)
    if seed is not None:
        seed = int(md5(str(seed).encode()).hexdigest(), 16) % 10**4
        tinyphysics.np.random.seed(seed)
    return sim.rollout(), sim.target_lataccel_history, sim.current_lataccel_history


def run_genome(genome, seg, model_path=MODEL_PATH, data_dir=DATA_PATH, debug=False, seed=None):
    data_dir = Path(data_dir)
    controller = pid.PID4Controller(*genome)
    file = data_dir / f"{seg:05d}.csv"
    # normalize path to the posix path
    if not file.is_absolute():
        file = PurePosixPath(file)
    return run_rollout(file, controller, model_path, debug=debug, seed=seed)


def run_bundle(bundle, model_path=MODEL_PATH, debug=False):
    match bundle:
        case genome, seg:
            return run_genome(genome, seg, model_path=model_path, debug=debug)
        case genome, seg, seed:
            return run_genome(genome, seg, model_path=model_path, debug=debug, seed=seed)


def test_genome(genome, segs, model_path=MODEL_PATH, data_dir=DATA_PATH, max_workers=None, tqdm=True, **kwargs):
    data_dir = Path(data_dir)
    run_rollout_partial = partial(run_bundle, model_path=model_path, debug=False)
    bundles = [(genome, seg) for seg in segs]
    if max_workers is None or max_workers > 1:
        if tqdm:
            results = process_map(run_rollout_partial, bundles, max_workers=max_workers, **kwargs)
        else:
            results = Pool(processes=max_workers).map(run_rollout_partial, bundles, **kwargs)
    else:
        if tqdm:
            results = [run_rollout_partial(bundle) for bundle in tqdm(bundles)]
        else:
            results = [run_rollout_partial(bundle) for bundle in bundles]

    # results = [run_rollout_partial(bundle) for bundle in bundles]
    costs = [result[0] for result in results]
    return costs


files_to_test = [random.randrange(0, 20000 - 1) for _ in range(10)]
genome = (0.12, 0.115, 0.005, -0.0008)
# test_genome(genome, files_to_test)
# run_bundle((genome, files_to_test[0]))
