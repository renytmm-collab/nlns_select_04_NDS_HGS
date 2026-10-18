"""Deterministic Python CVRP mechanics. No learned or reference inputs."""
import math

from .contracts import require, validate_instance, validate_routes


def distances(instance):
    xy = instance["coordinates"]
    return [[math.dist(a, b) for b in xy] for a in xy]


def route_cost(routes, distance):
    return sum(distance[a][b] for r in routes for a, b in zip([0] + r, r + [0]))


def action_mask(customers, selected):
    require(all(type(x) is int and 1 <= x <= customers for x in selected), "illegal action")
    require(len(selected) == len(set(selected)), "repeated action")
    chosen = set(selected)
    return [i == 0 or i in chosen for i in range(customers + 1)]


def insert_sequence(instance, partial, sequence, distance=None):
    """Existing feasible routes take precedence over opening a new route."""
    distance = distances(instance) if distance is None else distance
    routes = [list(r) for r in partial if r]
    loads = [sum(instance["demands"][i] for i in r) for r in routes]
    for node in sequence:
        best = None
        for ri, route in enumerate(routes):
            if loads[ri] + instance["demands"][node] > instance["capacity"]:
                continue
            for pos in range(len(route) + 1):
                a = route[pos - 1] if pos else 0
                b = route[pos] if pos < len(route) else 0
                candidate = (distance[a][node] + distance[node][b] - distance[a][b], ri, pos)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            routes.append([node])
            loads.append(instance["demands"][node])
        else:
            _, ri, pos = best
            routes[ri].insert(pos, node)
            loads[ri] += instance["demands"][node]
    return routes


def initial_solution(instance):
    validate_instance(instance)
    routes = insert_sequence(instance, [], list(range(1, len(instance["demands"]))))
    validate_routes(instance, routes)
    return routes


def repair(instance, routes, sequence, distance=None):
    validate_routes(instance, routes)
    require(bool(sequence), "empty deletion sequence")
    action_mask(len(instance["demands"]) - 1, sequence)
    removed = set(sequence)
    partial = [[i for i in r if i not in removed] for r in routes]
    result = insert_sequence(instance, partial, sequence, distance)
    validate_routes(instance, result)
    return result


def ordinary_state(instance, routes):
    """Whitelist projection: no metadata, IDs, split, paths or teacher fields."""
    validate_routes(instance, routes)
    return {"coordinates": instance["coordinates"], "demands": instance["demands"],
            "capacity": instance["capacity"], "routes": routes}


def leave_one_out(rewards):
    require(len(rewards) >= 2, "at least two independent rollouts required")
    total = sum(rewards)
    return [r - (total - r) / (len(rewards) - 1) for r in rewards]
