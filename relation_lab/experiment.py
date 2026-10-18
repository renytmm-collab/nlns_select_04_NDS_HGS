"""Experiment CLI. All scientific execution is gated; preflight is read-only."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

from .contracts import digest, require
from .runtime import (Run, dependency_status, file_hash, gate, identities, labels_for,
                      read_json, validate_config, validate_inputs, write_json)


def parser():
    p = argparse.ArgumentParser(description="Independent autoregressive destroy + greedy repair experiment")
    p.add_argument("command", choices=["preflight", "make-smoke", "solver-identity", "generate-references", "train", "resume", "export", "deploy", "evaluate", "freeze", "relation-diagnostics", "compare", "benchmark"])
    p.add_argument("--task", choices=["train", "evaluate", "generate-references"], default="train")
    for key in ("config", "data", "states", "references", "run-dir", "checkpoint", "policy", "freeze", "validation-results"):
        p.add_argument("--" + key)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--confirm-real-run", action="store_true")
    p.add_argument("--stop-after", type=int)
    p.add_argument("--left", nargs="+")
    p.add_argument("--right", nargs="+")
    return p


def required(args, *keys):
    for key in keys:
        require(getattr(args, key) is not None, "--" + key.replace("_", "-") + " required")


def load_inputs(args, task):
    required(args, "config", "data")
    c, data = read_json(args.config), read_json(args.data)
    states = read_json(args.states) if args.states else None
    require(states is not None or task == "generate-references", "--states required")
    validate_inputs(c, data, states)
    refs, labels = None, {}
    if task == "train" and c["lambda"] > 0:
        required(args, "references")
        refs = read_json(args.references)
        labels = labels_for(c, data, refs)
    else:
        require(args.references is None, "reference argument forbidden for lambda=0/evaluation/deployment/generation")
    return c, data, states, refs, labels


def diagnose(c, data, states, labels, checkpoint_path, run):
    import torch
    from .neural import Model
    from .vrp import ordinary_state
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    require(checkpoint["identity"] == run.manifest["identity"], "diagnostic checkpoint identity mismatch")
    model = Model(**checkpoint["spec"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    density = sum(sum(sum(r) for r in label["p"]) / (len(label["p"])*(len(label["p"])-1)) for label in labels.values()) / len(labels)
    constant = min(1-1e-6, max(1e-6, density))
    rows = []
    for inst in data:
        if inst["split"] != "train":
            continue
        target = torch.tensor(labels[inst["id"]]["p"])
        n = len(target)
        ix = torch.triu_indices(n, n, 1)
        values = []
        for s in states:
            if s["instance_id"] != inst["id"]:
                continue
            with torch.no_grad():
                logits = model.pair_logits(model.encoder(ordinary_state(inst, s["routes"])))
            y, x = target[ix[0], ix[1]], logits[ix[0], ix[1]]
            adjacency = torch.full_like(target, .1)
            for route in s["routes"]:
                for a, b in zip(route, route[1:]):
                    adjacency[a-1, b-1] = adjacency[b-1, a-1] = .9
            values.append({"bce": float(torch.nn.functional.binary_cross_entropy_with_logits(x, y)),
                           "brier": float(((x.sigmoid()-y)**2).mean()),
                           "constant_bce": float(torch.nn.functional.binary_cross_entropy(torch.full_like(y, constant), y)),
                           "current_adjacency_bce": float(torch.nn.functional.binary_cross_entropy(adjacency[ix[0], ix[1]], y))})
        row = {"instance_id": inst["id"], "states": len(values), "target_density": float(target[ix[0], ix[1]].mean()),
               **{k: sum(v[k] for v in values)/len(values) for k in values[0]}}
        rows.append(row)
        run.metric(row)
    write_json(run.path / "diagnostics.json", {"split": "train", "targets": "true_relations_even_for_shuffle", "constant_fitted_on_training_labels": constant, "rows": rows})


def main(argv=None):
    args = parser().parse_args(argv)
    run = None
    try:
        if args.command == "solver-identity":
            from .references import solver_identity
            print(json.dumps(solver_identity()))
            return 0
        required(args, "run_dir") if args.command not in ("preflight",) else None
        if args.command == "make-smoke":
            from .fixtures import smoke_bundle
            c, data, states, refs = smoke_bundle()
            run = Run(args.run_dir, args.command, identities(c, data, states, refs), c)
            for group, weight in (("baseline", 0.), ("relation", .1), ("shuffled_relation", .1)):
                variant = copy.deepcopy(c)
                variant.update(group=group)
                variant["lambda"] = weight
                write_json(run.path / (group+".json"), variant)
            for name, value in (("data", data), ("states", states), ("references", refs)):
                write_json(run.path / (name+".json"), value)
        elif args.command == "benchmark":
            from .fixtures import smoke_bundle
            from .vrp import repair
            c, data, states, _ = smoke_bundle()
            run = Run(args.run_dir, args.command, identities(c, data, states, None), c)
            start = time.perf_counter()
            for _ in range(200):
                repair(data[0], states[0]["routes"], [2, 5, 1])
            elapsed = time.perf_counter()-start
            result = {"synthetic_only": True, "customers": 8, "repairs": 200, "seconds": elapsed,
                      "repairs_per_second": 200/elapsed, "includes_validation_and_distance_construction": True,
                      "neural_throughput": "not_measured", "cvrp100_throughput": "not_measured"}
            write_json(run.path / "benchmark.json", result)
            print(json.dumps(result))
        elif args.command == "compare":
            from .evaluation import paired_report
            required(args, "left", "right")
            left, right = [read_json(p) for p in args.left], [read_json(p) for p in args.right]
            result = paired_report(left, right)
            run = Run(args.run_dir, args.command, {"left": [file_hash(p) for p in args.left], "right": [file_hash(p) for p in args.right]}, {})
            write_json(run.path / "paired.json", result)
        elif args.command == "export":
            required(args, "checkpoint")
            from .evaluation import export_checkpoint
            run = Run(args.run_dir, args.command, {"checkpoint": file_hash(args.checkpoint)}, {})
            export_checkpoint(args.checkpoint, run)
        else:
            task = args.task if args.command == "preflight" else ("train" if args.command in ("train", "resume", "relation-diagnostics") else args.command)
            c, data, states, refs, labels = load_inputs(args, task)
            identity = identities(c, data, states, refs)
            if args.command == "preflight":
                if task == "train":
                    require(sum(i["split"] == "train" for i in data) >= c["training"]["batch_size"], "insufficient training instances")
                if task == "generate-references":
                    validate_generation(c)
                print(json.dumps({"contract_valid": True, "identity": identity, **dependency_status(),
                                  "execution_requires_confirmation": c["kind"] == "experiment",
                                  "budgets_confirmed": c.get("budgets_confirmed", False),
                                  "note": "preflight does not run solver, training or inference"}))
                return 0
            if args.command == "freeze":
                required(args, "policy", "validation_results")
                validation = read_json(args.validation_results)
                require(validation["split"] == "validation" and validation["config_hash"] == digest(c) and validation["policy_hash"] == file_hash(args.policy), "freeze requires matching validation solver results")
                require(validation["protocol"]["data"] == digest(data) and validation["protocol"]["states"] == digest(states), "validation dataset identity mismatch")
                run = Run(args.run_dir, args.command, identity, c)
                write_json(run.path / "freeze.json", {"config_hash": digest(c), "data_hash": digest(data), "states_hash": digest(states),
                    "policy_hash": file_hash(args.policy), "selection": "prespecified last training step; configuration explicitly chosen before test",
                    "validation_results_hash": file_hash(args.validation_results)})
            else:
                gate(c, args.synthetic, args.confirm_real_run)
                if args.command == "generate-references":
                    from .references import generate
                    validate_generation(c)
                    run = Run(args.run_dir, args.command, identity, c)
                    generated = {}
                    for inst in data:
                        if inst["split"] == "train":
                            generated[inst["id"]] = generate(inst, c["reference"], c["reference_generation"]["max_2opt_moves_per_start"])
                            write_json(run.path / ("reference-"+digest(inst)+".json"), generated[inst["id"]])
                    require(generated, "training split empty")
                    write_json(run.path / "references.json", generated)
                elif args.command in ("train", "resume"):
                    if args.command == "resume":
                        required(args, "checkpoint")
                    else:
                        require(args.checkpoint is None, "use resume command for checkpoint restore")
                    from .training import train
                    run = Run(args.run_dir, args.command, identity, c, parent=file_hash(args.checkpoint) if args.checkpoint else None)
                    result = train(c, data, states, labels, identity, run, args.checkpoint, args.stop_after)
                    print(json.dumps(result))
                    return 0
                elif args.command == "evaluate":
                    required(args, "policy")
                    from .evaluation import evaluate
                    frozen = read_json(args.freeze) if args.freeze else None
                    run = Run(args.run_dir, args.command, identity, c)
                    evaluate(c, data, states, args.policy, args.split, run, frozen)
                elif args.command == "deploy":
                    required(args, "policy")
                    import torch
                    from .evaluation import load_policy, rng_for
                    from .vrp import ordinary_state, repair
                    policy, _ = load_policy(args.policy, c)
                    run = Run(args.run_dir, args.command, identity, c)
                    ids = {i["id"]: i for i in data}
                    for state in states:
                        inst = ids[state["instance_id"]]
                        rng = rng_for(c["seed"], state["instance_hash"], digest(state), "deploy", 0)
                        with torch.no_grad():
                            actions, _, _ = policy(ordinary_state(inst, state["routes"]), c["destroy_size"], rng)
                        run.metric({"instance_id": inst["id"], "actions": actions, "routes": repair(inst, state["routes"], actions)})
                elif args.command == "relation-diagnostics":
                    require(labels, "diagnostics require a positive-lambda training configuration")
                    required(args, "checkpoint")
                    run = Run(args.run_dir, args.command, identity, c)
                    diagnose(c, data, states, labels, args.checkpoint, run)
        run.finish()
        print(json.dumps({"status": "complete", "run_dir": str(run.path)}))
        return 0
    except Exception as exc:
        if run is not None:
            run.finish("failed", error=exc)
        print("FAILED: " + str(exc), file=sys.stderr)
        return 1


def validate_generation(c):
    from .references import solver_identity
    r = c["reference"]
    require(r["solver"] == solver_identity(), "reference generator identity mismatch; explicitly set solver-identity output")
    require(type(r["M"]) is int and r["M"] > 0 and len(r["seeds"]) == r["M"], "invalid generation M/seeds")
    require(all(type(s) is int and 0 <= s < 2**31 for s in r["seeds"]) and len(set(r["seeds"])) == len(r["seeds"]), "generation requires distinct integer seeds")
    require(r["duplicates"] in ("keep_equal_weight", "reject"), "explicit duplicate reference policy required")
    require(r["budget"]["kind"] == "iterations" and type(r["budget"]["limit"]) is int and r["budget"]["limit"] > 0, "explicit multistart budget required")
    moves = c["reference_generation"]["max_2opt_moves_per_start"]
    require(type(moves) is int and moves >= 0, "explicit 2opt cap required")
    if c["kind"] == "synthetic":
        require(r["M"] <= 3 and r["budget"]["limit"] <= 3 and moves <= 3, "synthetic reference budget exceeded")


if __name__ == "__main__":
    sys.exit(main())
