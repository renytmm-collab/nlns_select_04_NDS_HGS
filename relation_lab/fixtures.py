"""Small explicitly synthetic fixtures, never research collection."""
import random

from .contracts import instance_hash, validate_routes
from .vrp import insert_sequence


def smoke_bundle():
    data, states, refs = [], [], {}
    reference = {"M": 2, "duplicates": "keep_equal_weight", "seeds": [101, 102],
                 "budget": {"kind": "iterations", "limit": 1},
                 "solver": {"name": "synthetic-fixture", "version": "1", "binary_sha256": "0"*64}}
    for j, split in enumerate(["train", "train", "validation", "test"]):
        rng = random.Random(8123+j)
        inst = {"id": "smoke-%d" % j, "family_id": "smoke-%d" % j, "split": split, "synthetic": True,
                "distribution": "synthetic-uniform-eight-v1", "coordinates": [[rng.random(), rng.random()] for _ in range(9)],
                "demands": [0]+[1]*8, "capacity": 3, "distance": "euclidean_float64"}
        data.append(inst)
        # Deliberately nontrivial but poor fixed starting route structure.
        routes = [[1, 4, 7], [2, 5, 8], [3, 6]]
        states.append({"instance_id": inst["id"], "instance_hash": instance_hash(inst), "routes": routes})
        if split == "train":
            refs[inst["id"]] = []
            for seed in reference["seeds"]:
                order = list(range(1, 9))
                random.Random(seed).shuffle(order)
                solution = insert_sequence(inst, [], order)
                refs[inst["id"]].append({"instance_id": inst["id"], "instance_hash": instance_hash(inst), "status": "success",
                    "source": "synthetic fixture only", "solver": reference["solver"], "budget": reference["budget"],
                    "seed": seed, "routes": solution, "cost": validate_routes(inst, solution)})
    cfg = {"schema": 2, "kind": "synthetic", "budgets_confirmed": False, "distribution": "synthetic-uniform-eight-v1",
           "customers": 8, "distance": "euclidean_float64", "destroy_size": 3, "group": "baseline", "lambda": 0.,
           "seed": 31, "shuffle_seed": 73, "model": {"width": 16, "heads": 2, "layers": 1},
           "insertion": "action_order_existing_first", "acceptance": "strict_improvement", "reference": reference,
           "reference_generation": {"max_2opt_moves_per_start": 2},
           "training": {"steps": 2, "batch_size": 2, "rollouts": 3, "learning_rate": .001, "grad_clip": 1., "device": "cpu", "threads": 1, "checkpoint_rule": "last_step"},
           "evaluation": {"repeats": 2, "iterations": 3, "seconds": .05, "seeds": [51, 52]}}
    return cfg, data, states, refs
