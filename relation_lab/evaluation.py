"""Reference-free deployment and paired-instance evaluation."""
import math
import random
import statistics
import time
import platform
from pathlib import Path

from .contracts import digest, instance_hash, require
from .runtime import file_hash, write_json
from .search import search
from .vrp import distances, ordinary_state, repair, route_cost


def export_checkpoint(checkpoint_path, run):
    import torch
    from .neural import Model, export_policy
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = Model(**checkpoint["spec"])
    model.load_state_dict(checkpoint["model"])
    policy = export_policy(model)
    payload = {"policy_only": True, "spec": policy.spec, "policy": policy.state_dict(),
               "source_config_hash": checkpoint["identity"]["config"],
               "source_data_hash": checkpoint["identity"]["data"],
               "source_checkpoint_sha256": file_hash(checkpoint_path), "trained_step": checkpoint["step"]}
    with (run.path / "policy.pt").open("xb") as stream:
        torch.save(payload, stream)
    return payload


def load_policy(path, config):
    import torch
    from .neural import Policy
    from .training import configure_torch
    device = configure_torch(config)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload.get("policy_only") is True, "deploy/evaluate requires exported policy-only artifact")
    require(payload["source_config_hash"] == digest(config), "policy/config identity mismatch")
    require(payload["trained_step"] == config["training"]["steps"], "evaluation requires prespecified final checkpoint")
    policy = Policy(**payload["spec"]).to(device)
    policy.load_state_dict(payload["policy"])
    return policy.eval(), payload


def rng_for(seed, content, state, kind, repeat):
    import torch
    return torch.Generator(device="cpu").manual_seed(int(digest([seed, content, state, kind, repeat])[:15], 16))


def evaluate(config, data, states, policy_path, split, run, frozen=None):
    import torch
    policy, payload = load_policy(policy_path, config)
    runtime = {"torch": str(torch.__version__), "device": config["training"]["device"],
               "cuda": torch.version.cuda, "processor": platform.processor(), "platform": platform.platform(),
               "gpu": torch.cuda.get_device_name() if config["training"]["device"] == "cuda" else None}
    run.manifest["environment"].update(runtime)
    run.save()
    require(payload["source_data_hash"] == digest(data), "evaluation dataset differs from registered training split catalog")
    if split == "test":
        require(frozen is not None, "test requires explicit frozen selection artifact")
        require(frozen["config_hash"] == digest(config) and frozen["data_hash"] == digest(data) and frozen["states_hash"] == digest(states) and frozen["policy_hash"] == file_hash(policy_path), "test freeze identity mismatch")
    instances = {i["id"]: i for i in data if i["split"] == split}
    require(instances, "requested evaluation split empty")
    rows = []
    for state in states:
        if state["instance_id"] not in instances:
            continue
        inst = instances[state["instance_id"]]
        routes = state["routes"]
        hsh, shash = instance_hash(inst), digest(state)
        for seed in config["evaluation"]["seeds"]:
            for repeat in range(config["evaluation"]["repeats"]):
                generator = rng_for(seed, hsh, shash, "single", repeat)
                start = time.perf_counter()
                distance = distances(inst)
                with torch.no_grad():
                    actions, _, _ = policy(ordinary_state(inst, routes), config["destroy_size"], generator)
                rebuilt = repair(inst, routes, actions, distance)
                new, old = route_cost(rebuilt, distance), route_cost(routes, distance)
                elapsed = time.perf_counter() - start
                row = {"kind": "single", "instance_hash": hsh, "instance_id": inst["id"], "state_hash": shash,
                       "eval_seed": seed, "repeat": repeat, "cost": new, "initial_cost": old,
                       "gain": old-new, "relative_gain": (old-new)/max(old, 1e-12),
                       "improved": new < old, "seconds": elapsed, "actions": actions}
                rows.append(row)
                run.metric(row)
            for mode in ("iterations", "time"):
                generator = rng_for(seed, hsh, shash, mode, 0)
                def choose(current):
                    with torch.no_grad():
                        actions, _, _ = policy(ordinary_state(inst, current), config["destroy_size"], generator)
                    return actions
                result = search(inst, routes, choose, mode=mode, iterations=config["evaluation"]["iterations"], seconds=config["evaluation"]["seconds"])
                old = route_cost(routes, distances(inst))
                row = {"kind": "search_"+mode, "instance_hash": hsh, "instance_id": inst["id"], "state_hash": shash,
                       "eval_seed": seed, "repeat": 0, "initial_cost": old, "gain": old-result["cost"], **result}
                rows.append(row)
                run.metric(row)
    protocol = {"data": digest(data), "states": digest(states), "split": split,
                "implementation": run.manifest["identity"]["code"], "runtime": runtime,
                "evaluation": config["evaluation"], "destroy_size": config["destroy_size"],
                "model": config["model"], "training": config["training"],
                "acceptance": config["acceptance"], "insertion": config["insertion"]}
    results = {"schema": 1, "split": split, "protocol_hash": digest(protocol), "protocol": protocol,
               "config_hash": digest(config), "policy_hash": file_hash(policy_path), "train_seed": config["seed"],
               "group": config["group"], "lambda": config["lambda"], "rows": rows,
               "summary": summarize(rows)}
    write_json(run.path / "results.json", results)
    return results


def summarize(rows):
    result = {}
    for kind in sorted({r["kind"] for r in rows}):
        grouped = {}
        for row in rows:
            if row["kind"] == kind:
                grouped.setdefault(row["instance_hash"], []).append(row)
        means = [statistics.mean(r["gain"] for r in group) for group in grouped.values()]
        result[kind] = {"instances": len(grouped), "mean_instance_gain": statistics.mean(means),
                        "instance_gain_distribution": sorted(means),
                        "mean_instance_improvement_probability": statistics.mean(statistics.mean(float(r["gain"] > 0) for r in group) for group in grouped.values())}
    return result


def paired_report(left, right, bootstrap_seed=123, bootstrap_samples=2000):
    require(left and right, "both sides required")
    require(type(bootstrap_samples) is int and bootstrap_samples > 0, "positive bootstrap sample count required")
    for side in (left, right):
        require(len({(r.get("group"), r.get("lambda")) for r in side}) == 1, "each comparison side must contain one group/lambda")
    require(len({r["protocol_hash"] for r in left+right}) == 1, "paired evaluation protocols differ")
    def index(results):
        indexed = {}
        seeds = set()
        for result in results:
            seed = result["train_seed"]
            require(seed not in seeds, "duplicate train seed result")
            seeds.add(seed)
            for r in result["rows"]:
                key = (seed, r["instance_hash"], r["state_hash"], r["kind"], r["eval_seed"], r["repeat"])
                require(key not in indexed, "duplicate evaluation row")
                indexed[key] = r
        return indexed
    a, b = index(left), index(right)
    require(a.keys() == b.keys(), "paired rows/seeds/states missing; no intersection-only reporting")
    report = {}
    for kind in sorted({key[3] for key in a}):
        per_instance, per_seed = {}, {}
        for key, row in a.items():
            if key[3] != kind:
                continue
            require(row["initial_cost"] == b[key]["initial_cost"], "initial costs differ")
            diff = b[key]["cost"] - row["cost"]  # positive means left is better
            per_instance.setdefault(key[1], []).append(diff)
            per_seed.setdefault(key[0], {}).setdefault(key[1], []).append(diff)
        values = [statistics.mean(v) for v in per_instance.values()]
        rng = random.Random(bootstrap_seed)
        samples = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(bootstrap_samples))
        report[kind] = {"positive_favors": "left", "instances": len(values), "paired_mean_cost_difference": statistics.mean(values),
                        "instance_bootstrap_95_percent": [samples[int(.025*bootstrap_samples)], samples[min(bootstrap_samples-1, int(.975*bootstrap_samples))]],
                        "per_instance": {k: statistics.mean(v) for k, v in per_instance.items()},
                        "per_train_seed": {str(s): statistics.mean(statistics.mean(v) for v in instances.values()) for s, instances in per_seed.items()},
                        "limitation": "CI resamples instances after averaging supplied seeds; seed uncertainty and configuration selection are not covered"}
    return report
