"""Optional modest teacher: multistart cheapest insertion + bounded 2-opt.

Not HGS and not guaranteed high quality. Import externally qualified references
for the main experiment; generated reference quality must be audited separately.
"""
import random
import time
from pathlib import Path

from .contracts import instance_hash, require, validate_routes, build_relations
from .runtime import file_hash
from .vrp import distances, insert_sequence, route_cost


def solver_identity():
    from . import vrp
    from .contracts import digest
    return {"name": "python_multistart_insertion_2opt", "version": "1",
            "binary_sha256": digest({"references.py": file_hash(__file__), "vrp.py": file_hash(vrp.__file__)})}


def two_opt(routes, distance, max_moves):
    routes = [r[:] for r in routes]
    for _ in range(max_moves):
        best = None
        for ri, route in enumerate(routes):
            for i in range(len(route)-1):
                for j in range(i+1, len(route)):
                    a, b = (route[i-1] if i else 0), route[i]
                    c, d = route[j], (route[j+1] if j+1 < len(route) else 0)
                    delta = distance[a][c] + distance[b][d] - distance[a][b] - distance[c][d]
                    candidate = (delta, ri, i, j)
                    if delta < -1e-12 and (best is None or candidate < best):
                        best = candidate
        if best is None:
            break
        _, ri, i, j = best
        routes[ri][i:j+1] = reversed(routes[ri][i:j+1])
    return routes


def generate(instance, cfg, max_2opt_moves):
    require(instance["split"] == "train", "reference generation is training-only")
    require(cfg["solver"] == solver_identity(), "generator identity mismatch; run solver-identity and explicitly update config")
    require(cfg["budget"]["kind"] == "iterations" and type(cfg["budget"]["limit"]) is int and cfg["budget"]["limit"] > 0, "generator budget counts exact multistarts")
    require(type(max_2opt_moves) is int and max_2opt_moves >= 0, "explicit 2opt move cap required")
    require(len(cfg["seeds"]) == cfg["M"], "reference seed count mismatch")
    distance, refs = distances(instance), []
    for seed in cfg["seeds"]:
        rng, best, best_cost = random.Random(seed), None, float("inf")
        start = time.perf_counter()
        for _ in range(cfg["budget"]["limit"]):
            order = list(range(1, len(instance["demands"])))
            rng.shuffle(order)
            candidate = two_opt(insert_sequence(instance, [], order, distance), distance, max_2opt_moves)
            cost = route_cost(candidate, distance)
            if cost < best_cost:
                best, best_cost = candidate, cost
        refs.append({"instance_id": instance["id"], "instance_hash": instance_hash(instance),
                     "status": "success", "source": "independent multistart insertion + intra-route 2opt; quality uncertified",
                     "solver": cfg["solver"], "budget": cfg["budget"], "seed": seed,
                     "routes": best, "cost": validate_routes(instance, best),
                     "actual_seconds": time.perf_counter()-start, "max_2opt_moves_per_start": max_2opt_moves,
                     "starts_completed": cfg["budget"]["limit"]})
    build_relations(instance, refs, cfg)
    return refs
