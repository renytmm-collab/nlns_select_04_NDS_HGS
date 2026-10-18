"""Online search, injectable policy, explicit deadline semantics."""
import time
import math

from .contracts import require, validate_routes
from .vrp import distances, repair, route_cost


def search(instance, initial, choose, *, mode, iterations=None, seconds=None, clock=time.perf_counter):
    require(mode in ("iterations", "time"), "unknown stopping rule")
    require((type(iterations) is int and iterations > 0) if mode == "iterations" else (type(seconds) in (int, float) and math.isfinite(seconds) and seconds > 0), "positive explicit search budget required")
    start = clock()
    validate_routes(instance, initial)
    d = distances(instance)
    current = [r[:] for r in initial]
    cost = route_cost(current, d)
    trace = [{"seconds": 0.0, "iteration": 0, "cost": cost}]
    attempted = completed = 0
    late = False
    while (attempted < iterations) if mode == "iterations" else (clock() - start < seconds):
        actions = choose(current)
        candidate = repair(instance, current, actions, d)
        candidate_cost = route_cost(candidate, d)
        attempted += 1
        elapsed = clock() - start
        if mode == "time" and elapsed > seconds:
            late = True
            break  # Never credit a candidate that finished after the budget.
        completed += 1
        if candidate_cost < cost:
            current, cost = candidate, candidate_cost
            trace.append({"seconds": elapsed, "iteration": completed, "cost": cost})
    elapsed = clock() - start
    return {"routes": current, "cost": cost, "elapsed_seconds": elapsed, "attempted": attempted,
            "completed": completed, "late_candidate_discarded": late,
            "overshoot_seconds": max(0., elapsed-seconds) if mode == "time" else 0.,
            "trace": trace, "proposals_per_second": attempted / max(elapsed, 1e-12)}
