"""Convert CVRPLIB XML100 (Queiroga et al. 2021) into this project's DATA/STATES/REFERENCES/configs.

Standard library only; no download. Get instances.7z and solutions.7z from
https://galgos.inf.puc-rio.br/cvrplib/index.php/en/xml100 , extract both under one directory
(7z x instances.7z; 7z x solutions.7z), then:

  python -B scripts/prepare_xml100.py --out-dir runs/xml100_v1 --src-dir /path/to/xml100

What it does (all deterministic given --sample-seed):
  * samples disjoint train/validation/test instances from the zip (each instance is its own family);
  * writes data.json with coordinates, demands, capacity, Euclidean float64 distance;
  * writes states.json: one deterministic customer-order greedy start per instance, plus
    --train-extra-states random-order greedy starts for train instances only. The starting
    rule never looks at reference solutions;
  * writes references.json (M=1): the published optimal solution of each TRAIN instance.
    The published solutions are optimal for the dataset's rounded distances, not for the
    float distances used here, so they are strong but not exactly optimal references;
  * writes baseline/relation/shuffled_relation configs from configs/experiment_template.json
    with budgets_confirmed=false. Flip it yourself after a throughput check.
"""
import argparse
import hashlib
import json
import math
import random
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from relation_lab.contracts import instance_hash, validate_routes  # noqa: E402
from relation_lab.runtime import read_json, write_json  # noqa: E402
from relation_lab.vrp import initial_solution, insert_sequence  # noqa: E402

DISTRIBUTION = "XML100-mixed-sample-v1"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Source:
    """Paired <stem>.vrp / <stem>.sol files from a zip or an already extracted directory."""

    def __init__(self, zip_path=None, directory=None):
        self.archive = zipfile.ZipFile(zip_path) if zip_path else None
        if self.archive:
            names = self.archive.namelist()
        else:
            names = [str(p) for p in Path(directory).rglob("*") if p.is_file()]
        self.vrp, self.sol = {}, {}
        for name in names:
            path = Path(name)
            if path.suffix == ".vrp":
                self.vrp[path.stem] = name
            elif path.suffix == ".sol":
                self.sol[path.stem] = name

    def stems(self):
        return sorted(set(self.vrp) & set(self.sol))

    def read(self, name):
        if self.archive:
            return self.archive.read(name).decode("utf-8")
        return Path(name).read_text(encoding="utf-8")

    def identity(self, zip_path):
        if zip_path:
            return sha256_file(zip_path)
        digest = hashlib.sha256()  # hash of content of every used file, in stem order
        for stem in self.stems():
            digest.update(stem.encode())
            digest.update(Path(self.vrp[stem]).read_bytes())
            digest.update(Path(self.sol[stem]).read_bytes())
        return digest.hexdigest()


def parse_vrp(text, name):
    header, section, coords, demands, depots = {}, None, {}, {}, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line == "EOF":
            continue
        if re.fullmatch(r"[A-Z_]+_SECTION", line):
            section = line
            continue
        if ":" in line and not line[0].isdigit() and line[0] != "-":
            key, value = line.split(":", 1)
            header[key.strip().upper()] = value.strip()
            section = None
            continue
        tokens = line.split()
        if section == "NODE_COORD_SECTION":
            coords[int(tokens[0])] = (float(tokens[1]), float(tokens[2]))
        elif section == "DEMAND_SECTION":
            demands[int(tokens[0])] = float(tokens[1])
        elif section == "DEPOT_SECTION":
            depots.extend(int(t) for t in tokens if int(t) != -1)
    ids = sorted(coords)
    dimension = int(header["DIMENSION"])
    if header.get("TYPE", "CVRP").upper() != "CVRP" or len(ids) != dimension or ids != list(range(1, dimension + 1)):
        raise ValueError("%s: unexpected TYPE/DIMENSION/node numbering" % name)
    if depots != [1] or set(demands) != set(ids) or demands[1] != 0:
        raise ValueError("%s: expected single depot id 1 with zero demand" % name)
    return {"coordinates": [list(coords[i]) for i in ids], "demands": [demands[i] for i in ids],
            "capacity": float(header["CAPACITY"]), "edge_weight_type": header.get("EDGE_WEIGHT_TYPE")}


def parse_sol(text, name):
    routes, cost = [], None
    for line in text.splitlines():
        m = re.match(r"\s*Route\s*#?\s*\d+\s*:\s*(.*)", line, re.I)
        if m and m.group(1).split():
            routes.append([int(t) for t in m.group(1).split()])
            continue
        m = re.match(r"\s*Cost\s*:?\s*([0-9.eE+-]+)", line, re.I)
        if m:
            cost = float(m.group(1))
    if not routes or cost is None:
        raise ValueError("%s: no routes or cost in solution file" % name)
    return routes, cost


def rounded_cost(instance, routes):
    xy = instance["coordinates"]
    total = 0
    for route in routes:
        path = [0] + route + [0]
        total += sum(int(math.floor(math.dist(xy[a], xy[b]) + 0.5)) for a, b in zip(path, path[1:]))
    return total


def random_order_state(instance, key):
    order = list(range(1, len(instance["demands"])))
    random.Random(key).shuffle(order)
    return insert_sequence(instance, [], order)


def published_solution_valid(source, stem):
    """False when files cannot be parsed or the solution repeats/omits a customer or breaks capacity."""
    try:
        parsed = parse_vrp(source.read(source.vrp[stem]), stem)
        routes, _ = parse_sol(source.read(source.sol[stem]), stem)
        n = len(parsed["demands"]) - 1
        if sorted(v for r in routes for v in r) != list(range(1, n + 1)):
            return False
        return all(sum(parsed["demands"][v] for v in r) <= parsed["capacity"] for r in routes)
    except (ValueError, KeyError, IndexError):
        return False


def build(args, source, archive_hash, out):
    template = read_json(ROOT / "configs" / "experiment_template.json")
    if True:
        paired = source.stems()
        stems, excluded = [], []
        for stem in paired:  # only instances whose published solution is a valid CVRP solution
            (stems if published_solution_valid(source, stem) else excluded).append(stem)
        total = args.train + args.validation + args.test
        if len(stems) < total:
            raise SystemExit("source has only %d valid paired .vrp/.sol instances (%d excluded), need %d" % (len(stems), len(excluded), total))
        chosen = random.Random(args.sample_seed).sample(stems, total)
        plan = ([("train", s) for s in chosen[:args.train]]
                + [("validation", s) for s in chosen[args.train:args.train + args.validation]]
                + [("test", s) for s in chosen[args.train + args.validation:]])
        solver = {"name": "CVRPLIB-XML100-published-optimal", "version": "Queiroga-et-al-2021",
                  "binary_sha256": archive_hash}
        budget = {"kind": "iterations", "limit": 1}
        data, states, references, mismatches, customers = [], [], {}, 0, None
        for split, stem in plan:
            parsed = parse_vrp(source.read(source.vrp[stem]), stem)
            n = len(parsed["demands"]) - 1
            if customers is None:
                customers = n
            if n != customers or (args.customers and n != args.customers):
                raise SystemExit("%s has %d customers; expected a single size (%s)" % (stem, n, args.customers or customers))
            inst = {"id": stem, "family_id": stem, "split": split, "distribution": DISTRIBUTION,
                    "coordinates": parsed["coordinates"], "demands": parsed["demands"],
                    "capacity": parsed["capacity"], "distance": "euclidean_float64"}
            data.append(inst)
            ih = instance_hash(inst)
            seen = set()
            start_routes = [initial_solution(inst)]
            if split == "train":
                start_routes += [random_order_state(inst, "%s:%s:%d" % (args.sample_seed, stem, k))
                                 for k in range(args.train_extra_states)]
            for routes in start_routes:
                key = tuple(tuple(r) for r in routes)
                if key not in seen:
                    seen.add(key)
                    states.append({"instance_id": stem, "instance_hash": ih, "routes": routes})
            if split == "train":
                routes, file_cost = parse_sol(source.read(source.sol[stem]), stem)
                cost = validate_routes(inst, routes)  # raises if node mapping or capacity is wrong
                if parsed["edge_weight_type"] == "EUC_2D" and rounded_cost(inst, routes) != round(file_cost):
                    mismatches += 1
                references[stem] = [{"instance_id": stem, "instance_hash": ih, "status": "success",
                                     "source": "CVRPLIB XML100 published optimal solution file %s.sol; budget is a placeholder" % stem,
                                     "solver": solver, "budget": budget, "seed": 101, "routes": routes, "cost": cost}]
        if args.train and mismatches / args.train > args.max_cost_mismatch_fraction:
            raise SystemExit("rounded-cost mismatch on %d/%d solutions; wrong node mapping or distance convention" % (mismatches, args.train))
    base = json.loads(json.dumps(template))
    base.update(distribution=DISTRIBUTION, customers=customers, destroy_size=args.destroy_size,
                budgets_confirmed=False, seed=args.train_seed)
    base["reference"].update(M=1, seeds=[101], budget=budget, solver=solver)
    base["training"].update(steps=args.steps, device=args.device, threads=args.threads)
    base["evaluation"].update(repeats=args.repeats, iterations=args.iterations, seconds=args.seconds)
    for group, weight in (("baseline", 0.0), ("relation", args.lambda_), ("shuffled_relation", args.lambda_)):
        config = json.loads(json.dumps(base))
        config.update(group=group)
        config["lambda"] = weight
        write_json(out / (group + ".json"), config)
    write_json(out / "data.json", data)
    write_json(out / "states.json", states)
    write_json(out / "references.json", references)
    summary = {"archive_sha256": archive_hash, "customers": customers, "sample_seed": args.sample_seed,
               "instances": {s: sum(d["split"] == s for d in data) for s in ("train", "validation", "test")},
               "states": len(states), "reference_rounded_cost_mismatches": mismatches,
               "paired_instances": len(paired), "excluded_invalid_published_solution": len(excluded),
               "excluded_ids": excluded,
               "ids": {s: [d["id"] for d in data if d["split"] == s] for s in ("train", "validation", "test")},
               "note": "budgets_confirmed=false; set true only after measuring throughput"}
    write_json(out / "prepare_manifest.json", summary)
    return summary


def require_new(out):
    if out.exists():
        raise SystemExit("out-dir already exists: " + str(out))
    out.mkdir(parents=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True, help="new directory; must not exist")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--src-dir", help="directory holding extracted XML100 .vrp and .sol files (searched recursively)")
    group.add_argument("--zip", help="zip containing XML100 .vrp and .sol files")
    p.add_argument("--train", type=int, default=200)
    p.add_argument("--validation", type=int, default=50)
    p.add_argument("--test", type=int, default=100)
    p.add_argument("--customers", type=int, default=100, help="0 disables the size assertion")
    p.add_argument("--sample-seed", type=int, default=2026)
    p.add_argument("--train-extra-states", type=int, default=3)
    p.add_argument("--destroy-size", type=int, default=10)
    p.add_argument("--lambda", dest="lambda_", type=float, default=0.1)
    p.add_argument("--train-seed", type=int, default=31)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--seconds", type=float, default=2.0)
    p.add_argument("--max-cost-mismatch-fraction", type=float, default=0.02)
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    require_new(out)
    source = Source(zip_path=args.zip, directory=args.src_dir)
    summary = build(args, source, source.identity(args.zip), out)
    print(json.dumps({k: v for k, v in summary.items() if k != "ids"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
