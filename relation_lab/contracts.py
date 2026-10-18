import hashlib
import json
import math
import random


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def instance_hash(instance):
    """Ordered node content, excluding incidental IDs and split labels."""
    def numeric(value):
        value = float(value)
        return 0.0 if value == 0 else value
    return digest({"coordinates": [[numeric(v) for v in p] for p in instance["coordinates"]],
                   "demands": [numeric(v) for v in instance["demands"]],
                   "capacity": numeric(instance["capacity"]), "distance": instance["distance"]})


def validate_instance(instance):
    require(instance["distance"] == "euclidean_float64", "unsupported distance convention; explicit conversion required")
    xy, demand, capacity = instance["coordinates"], instance["demands"], instance["capacity"]
    require(len(xy) == len(demand) and len(xy) >= 2, "node dimensions")
    require(all(len(p) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in p) for p in xy), "invalid coordinates")
    require(isinstance(capacity, (int, float)) and not isinstance(capacity, bool) and math.isfinite(capacity) and capacity > 0, "invalid capacity")
    require(demand[0] == 0 and all(isinstance(d, (int, float)) and not isinstance(d, bool) and math.isfinite(d) and 0 <= d <= capacity for d in demand), "invalid demands")
    require(instance["split"] in ("train", "validation", "test"), "invalid split")
    require(isinstance(instance["id"], str) and instance["id"], "invalid instance ID")
    return instance_hash(instance)


def validate_dataset(instances):
    ids, hashes = set(), set()
    for inst in instances:
        h = validate_instance(inst)
        require(inst["id"] not in ids, "duplicate instance ID")
        require(h not in hashes, "duplicate instance content, including across splits")
        ids.add(inst["id"])
        hashes.add(h)
    require(bool(ids), "empty dataset")


def canonical_routes(routes):
    return tuple(sorted(min(tuple(r), tuple(reversed(r))) for r in routes))


def validate_routes(instance, routes):
    validate_instance(instance)
    n = len(instance["demands"])
    require(isinstance(routes, list) and routes and all(isinstance(r, list) and r for r in routes), "empty or malformed routes")
    nodes = [v for r in routes for v in r]
    require(all(type(v) is int and 1 <= v < n for v in nodes), "customer IDs must be 1..N; depot is implicit")
    require(sorted(nodes) == list(range(1, n)), "missing or duplicate customer")
    cost = 0.0
    for route in routes:
        require(sum(instance["demands"][v] for v in route) <= instance["capacity"], "capacity exceeded")
        path = [0] + route + [0]
        cost += sum(math.dist(instance["coordinates"][a], instance["coordinates"][b]) for a, b in zip(path, path[1:]))
    require(math.isfinite(cost), "nonfinite route cost")
    return cost


def build_relations(instance, references, config):
    validate_instance(instance)
    require(instance["split"] == "train", "references are training-only")
    require(type(config["M"]) is int and config["M"] > 0, "M must be positive")
    require(config["duplicates"] in ("keep_equal_weight", "reject"), "explicit duplicate policy required")
    require(len(references) == config["M"], "reference count mismatch; no retry or padding")
    require(len(config["seeds"]) == config["M"] and all(type(s) is int for s in config["seeds"]), "explicit seeds required")
    require(set(config["budget"]) == {"kind", "limit"} and config["budget"]["kind"] in ("seconds", "iterations") and type(config["budget"]["limit"]) in (int, float) and math.isfinite(config["budget"]["limit"]) and config["budget"]["limit"] > 0, "explicit positive budget required")
    identity = config["solver"]
    require(all(isinstance(identity.get(k), str) and identity[k] for k in ("name", "version", "binary_sha256")), "solver identity required")
    require(len(identity["binary_sha256"]) == 64 and all(c in "0123456789abcdef" for c in identity["binary_sha256"]), "invalid binary SHA256")
    n = len(instance["demands"]) - 1
    p = [[0.0] * n for _ in range(n)]
    seen, duplicates, costs = set(), 0, []
    for index, ref in enumerate(references):
        require(ref["instance_hash"] == instance_hash(instance) and ref["instance_id"] == instance["id"], "reference identity mismatch")
        require(ref["status"] == "success", "failed reference")
        require(ref["solver"] == identity and ref["budget"] == config["budget"] and ref["seed"] == config["seeds"][index], "reference provenance mismatch")
        require(isinstance(ref["source"], str) and ref["source"], "reference source required")
        cost = validate_routes(instance, ref["routes"])
        require(type(ref["cost"]) in (int, float) and math.isfinite(ref["cost"]) and math.isclose(cost, ref["cost"], rel_tol=1e-10, abs_tol=1e-10), "cost mismatch; printed solver cost is not authoritative")
        key = canonical_routes(ref["routes"])
        if key in seen:
            duplicates += 1
            require(config["duplicates"] != "reject", "duplicate reference")
        seen.add(key)
        costs.append(cost)
        for route in ref["routes"]:
            for a, b in zip(route, route[1:]):
                p[a - 1][b - 1] += 1
                p[b - 1][a - 1] += 1
    p = [[count / len(references) for count in row] for row in p]
    return {"instance_hash": instance_hash(instance), "p": p, "costs": costs, "duplicates": duplicates, "reference_hash": digest(references), "config_hash": digest(config)}


def shuffled_relation(p, content_hash, seed):
    """Fixed instance permutation, private RNG; independent of policy streams."""
    permutation = list(range(len(p)))
    random.Random(int(digest([content_hash, seed]), 16)).shuffle(permutation)
    return [[p[i][j] for j in permutation] for i in permutation], permutation


def relation_bce(logits, p):
    """All unordered customer pairs, equal weight, mean per state. Diagnostic only."""
    n = len(p)
    require(n >= 2 and len(logits) == n and all(len(r) == n for r in logits), "pair dimensions")
    losses = []
    for i in range(n):
        for j in range(i + 1, n):
            x, target = logits[i][j], p[i][j]
            require(math.isfinite(x) and x == logits[j][i], "finite symmetric logits required")
            require(0 <= target <= 1 and target == p[j][i], "symmetric soft targets required")
            losses.append(max(x, 0) - x * target + math.log1p(math.exp(-abs(x))))
    return sum(losses) / len(losses)
